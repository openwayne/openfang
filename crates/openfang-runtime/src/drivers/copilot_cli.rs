//! GitHub Copilot CLI backend driver.
//!
//! Spawns the `copilot` CLI (the standalone GitHub Copilot agent, **not** the
//! old `gh copilot` extension) to generate completions. Uses `-p/--prompt`
//! non-interactive mode with `--output-format json` for structured streaming.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use openfang_types::message::{ContentBlock, Role, StopReason, TokenUsage};
use serde::Deserialize;
use tokio::io::AsyncBufReadExt;
use tracing::{debug, info, warn};

/// Environment variable names to strip (same as Claude Code).
const SENSITIVE_ENV_EXACT: &[&str] = &[
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "OPENROUTER_API_KEY",
    "PERPLEXITY_API_KEY",
    "COHERE_API_KEY",
    "AI21_API_KEY",
    "CEREBRAS_API_KEY",
    "SAMBANOVA_API_KEY",
    "HUGGINGFACE_API_KEY",
    "XAI_API_KEY",
    "REPLICATE_API_TOKEN",
    "BRAVE_API_KEY",
    "TAVILY_API_KEY",
    "ELEVENLABS_API_KEY",
];

const SENSITIVE_SUFFIXES: &[&str] = &["_SECRET", "_TOKEN", "_PASSWORD"];

/// Events emitted by Copilot CLI in JSON mode.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CopilotEvent {
    #[serde(rename = "assistant.message_delta")]
    MessageDelta { data: MessageDeltaData },
    #[serde(rename = "assistant.reasoning_delta")]
    ReasoningDelta { data: ReasoningDeltaData },
    #[serde(rename = "assistant.message")]
    Message { data: MessageData },
    #[serde(rename = "tool.execution_start")]
    ToolStart { data: ToolStartData },
    #[serde(rename = "tool.execution_complete")]
    ToolComplete { data: ToolCompleteData },
    #[serde(rename = "result")]
    Result { usage: Option<CopilotUsage> },
    // Ignore other events
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    #[serde(rename = "deltaContent")]
    delta_content: String,
}

#[derive(Debug, Deserialize)]
struct ReasoningDeltaData {
    #[serde(rename = "deltaContent")]
    delta_content: String,
}

#[derive(Debug, Deserialize)]
struct MessageData {
    #[allow(dead_code)]
    content: String,
    #[serde(rename = "outputTokens")]
    output_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ToolStartData {
    #[serde(rename = "toolName")]
    tool_name: String,
}

#[derive(Debug, Deserialize)]
struct ToolCompleteData {
    #[serde(rename = "toolName")]
    tool_name: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct CopilotUsage {
    #[serde(rename = "inputTokens")]
    input_tokens: Option<u64>,
    #[serde(rename = "outputTokens")]
    output_tokens: Option<u64>,
    #[allow(dead_code)]
    #[serde(rename = "totalTokens")]
    total_tokens: Option<u64>,
}

/// LLM driver that delegates to the `copilot` CLI.
pub struct CopilotCliDriver {
    cli_path: String,
    /// When true, adds `--allow-all` (tools + paths + URLs) instead of only
    /// `--allow-all-tools`.  Mirrors the `skip_permissions` field in
    /// `DriverConfig` — safe because OpenFang's own RBAC layer restricts agents.
    skip_permissions: bool,
    /// Optional GitHub / Copilot token injected as `COPILOT_GITHUB_TOKEN`.
    api_key: Option<String>,
}

impl CopilotCliDriver {
    /// Create a new Copilot CLI driver.
    ///
    /// - `cli_path` overrides the CLI binary path; defaults to `"copilot"` on PATH.
    /// - `skip_permissions` maps to `--allow-all` (all tools/paths/URLs).
    /// - `api_key` is injected as the `COPILOT_GITHUB_TOKEN` environment variable.
    pub fn new(cli_path: Option<String>, skip_permissions: bool, api_key: Option<String>) -> Self {
        Self {
            cli_path: cli_path
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "copilot".to_string()),
            skip_permissions,
            api_key,
        }
    }

    /// Detect if the `copilot` CLI is available.
    pub fn detect() -> Option<String> {
        debug!("Attempting to detect copilot CLI...");
        let output = std::process::Command::new("copilot")
            .arg("--version")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    debug!("Copilot CLI detected: {}", version);
                    Some(version)
                } else {
                    debug!("Copilot CLI detected but returned error status: {:?}", output.status);
                    None
                }
            },
            Err(e) => {
                debug!("Copilot CLI detection failed: {}", e);
                None
            }
        }
    }

    /// Map a model ID like "copilot-cli/gpt-5.4" to CLI --model flag value.
    fn model_flag(model: &str) -> Option<String> {
        let stripped = model
            .strip_prefix("copilot-cli/")
            .unwrap_or(model);
        
        Some(stripped.to_string())
    }

    /// Build a text prompt from the completion request messages.
    fn build_prompt(request: &CompletionRequest) -> String {
        let mut parts = Vec::new();

        if let Some(ref sys) = request.system {
            parts.push(format!("[System]\n{sys}"));
        }

        for msg in &request.messages {
            let role_label = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            let text = msg.content.text_content();
            if !text.is_empty() {
                parts.push(format!("[{role_label}]\n{text}"));
            }
        }

        parts.join("\n\n")
    }

    /// Apply security env filtering to a command.
    fn apply_env_filter(cmd: &mut tokio::process::Command) {
        for key in SENSITIVE_ENV_EXACT {
            cmd.env_remove(key);
        }
        // Remove any env var with a sensitive suffix, unless it's GITHUB_*
        for (key, _) in std::env::vars() {
            if key.starts_with("GITHUB_") {
                continue;
            }
            let upper = key.to_uppercase();
            for suffix in SENSITIVE_SUFFIXES {
                if upper.ends_with(suffix) {
                    cmd.env_remove(&key);
                    break;
                }
            }
        }
    }
}

#[async_trait]
impl LlmDriver for CopilotCliDriver {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        // Implement complete via stream to reuse parsing logic
        let (tx, mut rx) = tokio::sync::mpsc::channel(100);
        
        let handle = tokio::spawn({
            let driver = CopilotCliDriver {
                cli_path: self.cli_path.clone(),
                skip_permissions: self.skip_permissions,
                api_key: self.api_key.clone(),
            };
            async move {
                driver.stream(request, tx).await
            }
        });

        // Drain the channel
        while rx.recv().await.is_some() {}

        handle.await.map_err(|e| LlmError::Http(format!("Join error: {e}")))?
    }
    
    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let prompt = Self::build_prompt(&request);
        let model_flag = Self::model_flag(&request.model);

        let mut cmd = tokio::process::Command::new(&self.cli_path);
        
        cmd.arg("-p")
            .arg(&prompt)
            .arg("--output-format")
            .arg("json")
            .arg("--stream")
            .arg("on")
            .arg("--no-ask-user")
            .arg("--no-custom-instructions")
            // Disable auto-update in daemon mode — non-interactive, should not
            // download binaries at runtime.
            .arg("--no-auto-update")
            // Suppress ANSI colour codes so JSON parsing is clean.
            .arg("--no-color")
            // Silence copilot's own internal log output so it doesn't leak
            // into stderr and obscure real error messages.
            .arg("--log-level")
            .arg("none");

        // Permission flags — daemon has no TTY, so prompts would block forever.
        // `--allow-all` grants tools + paths + URLs; `--allow-all-tools` is the
        // minimal grant required for non-interactive operation.
        if self.skip_permissions {
            cmd.arg("--allow-all");
        } else {
            cmd.arg("--allow-all-tools");
        }

        if let Some(ref model) = model_flag {
            cmd.arg("--model").arg(model);
        }

        // Map ThinkingConfig budget_tokens → Copilot reasoning-effort level.
        if let Some(ref thinking) = request.thinking {
            let level = match thinking.budget_tokens {
                0..=3_000 => "low",
                3_001..=8_000 => "medium",
                8_001..=15_000 => "high",
                _ => "xhigh",
            };
            cmd.arg("--reasoning-effort").arg(level);
        }

        Self::apply_env_filter(&mut cmd);

        // Inject GitHub / Copilot token so the CLI can authenticate without
        // relying on an interactive `gh auth login` session.
        if let Some(ref key) = self.api_key {
            cmd.env("COPILOT_GITHUB_TOKEN", key);
        }

        cmd.stdout(std::process::Stdio::piped());
        // Capture stderr to report errors
        cmd.stderr(std::process::Stdio::piped());

        info!(
            cli = %self.cli_path,
            model = ?model_flag,
            skip_permissions = self.skip_permissions,
            "Spawning copilot cli (streaming)"
        );

        let mut child = cmd
            .spawn()
            .map_err(|e| LlmError::Http(format!(
                "Copilot CLI not found or failed to start ({}). \
                 Install: npm install -g @githubnext/github-copilot-cli",
                e
            )))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LlmError::Http("No stdout from copilot CLI".to_string()))?;
        
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| LlmError::Http("No stderr from copilot CLI".to_string()))?;

        // Read stderr in background to prevent deadlock
        let stderr_handle = tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stderr);
            let mut output = String::new();
            let mut line = String::new();
            while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                output.push_str(&line);
                line.clear();
            }
            output
        });

        let reader = tokio::io::BufReader::new(stdout);
        let mut lines = reader.lines();

        let mut full_text = String::new();
        let mut final_usage = TokenUsage::default();

        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }

            // Parse JSON event
            match serde_json::from_str::<CopilotEvent>(&line) {
                Ok(event) => match event {
                    CopilotEvent::MessageDelta { data } => {
                        let text = data.delta_content;
                        full_text.push_str(&text);
                        let _ = tx.send(StreamEvent::TextDelta { text }).await;
                    }
                    CopilotEvent::ReasoningDelta { data } => {
                        let _ = tx.send(StreamEvent::ThinkingDelta { text: data.delta_content }).await;
                    }
                    CopilotEvent::Message { data } => {
                        // Final message content
                        // We already reconstructed it from deltas, but we can verify or use output tokens
                        if let Some(toks) = data.output_tokens {
                            final_usage.output_tokens = toks;
                        }
                    }
                    CopilotEvent::ToolStart { data } => {
                        // Report internal tool usage as thinking
                        let _ = tx.send(StreamEvent::ThinkingDelta { 
                            text: format!("\n[Executing tool: {}]\n", data.tool_name) 
                        }).await;
                    }
                    CopilotEvent::ToolComplete { data } => {
                        if let Some(name) = data.tool_name {
                             let _ = tx.send(StreamEvent::ThinkingDelta { 
                                text: format!("[Tool complete: {}]\n", name) 
                            }).await;
                        }
                    }
                    CopilotEvent::Result { usage } => {
                        if let Some(u) = usage {
                            if let Some(i) = u.input_tokens { final_usage.input_tokens = i; }
                            if let Some(o) = u.output_tokens { final_usage.output_tokens = o; }
                        }
                    }
                    CopilotEvent::Unknown => {
                        // Ignore
                    }
                },
                Err(e) => {
                    // Fallback for non-JSON lines (e.g. error messages or raw text)
                    // If parsing fails, it might be a raw error message from the CLI
                    warn!("Failed to parse Copilot CLI output line: {} (line: {})", e, line);
                }
            }
        }

        let status = child.wait().await.map_err(|e| LlmError::Http(format!("Copilot CLI wait failed: {e}")))?;
        let stderr_output = stderr_handle.await.unwrap_or_else(|_| "<stderr read failed>".to_string());

        if !status.success() {
             warn!(code = ?status.code(), stderr = %stderr_output, "Copilot CLI exited with error during stream");
             // If we didn't generate any content, this is a fatal error.
             if full_text.is_empty() {
                 return Err(LlmError::Api {
                     status: status.code().unwrap_or(1) as u16,
                     message: format!(
                        "Copilot CLI exited with error (code {}) and produced no output. \nStderr: {}", 
                        status.code().unwrap_or(1),
                        stderr_output.trim()
                     ),
                 });
             }
        }

        let _ = tx
            .send(StreamEvent::ContentComplete {
                stop_reason: StopReason::EndTurn,
                usage: final_usage,
            })
            .await;

        Ok(CompletionResponse {
            content: vec![ContentBlock::Text { text: full_text, provider_metadata: None }],
            stop_reason: StopReason::EndTurn,
            tool_calls: Vec::new(),
            usage: final_usage,
        })
    }
}

/// Check if the gh copilot CLI is available.
pub fn copilot_cli_available() -> bool {
    CopilotCliDriver::detect().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_simple() {
        use openfang_types::message::{Message, MessageContent};

        let request = CompletionRequest {
            model: "copilot-cli".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::text("Hello"),
            }],
            tools: vec![],
            max_tokens: 1024,
            temperature: 0.7,
            system: Some("You are helpful.".to_string()),
            thinking: None,
        };

        let prompt = CopilotCliDriver::build_prompt(&request);
        assert!(prompt.contains("[System]"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("[User]"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_new_defaults_to_copilot() {
        let driver = CopilotCliDriver::new(None, true, None);
        assert_eq!(driver.cli_path, "copilot");
        assert!(driver.skip_permissions);
        assert!(driver.api_key.is_none());
    }

    #[test]
    fn test_new_with_custom_path() {
        let driver = CopilotCliDriver::new(Some("/usr/local/bin/copilot".to_string()), false, None);
        assert_eq!(driver.cli_path, "/usr/local/bin/copilot");
        assert!(!driver.skip_permissions);
    }

    #[test]
    fn test_new_with_api_key() {
        let driver = CopilotCliDriver::new(None, true, Some("ghp_test".to_string()));
        assert_eq!(driver.api_key.as_deref(), Some("ghp_test"));
    }

    #[test]
    fn test_model_flag_mapping() {
        assert_eq!(
            CopilotCliDriver::model_flag("copilot-cli/gpt-5.4"),
            Some("gpt-5.4".to_string())
        );
        assert_eq!(
            CopilotCliDriver::model_flag("copilot-cli/claude-sonnet-4.6"),
            Some("claude-sonnet-4.6".to_string())
        );
        assert_eq!(
            CopilotCliDriver::model_flag("gpt-4.1"),
            Some("gpt-4.1".to_string())
        );
    }

    #[test]
    fn test_sensitive_env_list_coverage() {
        assert!(SENSITIVE_ENV_EXACT.contains(&"OPENAI_API_KEY"));
        // GITHUB_TOKEN is handled by prefix check
    }
}
