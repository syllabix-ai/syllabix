//! Eval-only MCP stdio server for the Riley card tools (#329).

fn main() -> std::process::ExitCode {
    let code = syllabix_riley_eval::mcp::run_from_env(
        |key| std::env::var(key).ok(),
        std::io::stdin().lock(),
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
    );
    std::process::ExitCode::from(code)
}
