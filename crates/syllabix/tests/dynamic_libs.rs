// Linux-only: the launch merge gate requires no undeclared dynamic runtime
// dependencies on the shipped executable. Later native links (ONNX, whisper,
// llama) must be added here explicitly or statically.

#[cfg(target_os = "linux")]
mod linux {
    use assert_cmd::cargo::cargo_bin;
    use std::process::Command;

    const ALLOWED_PREFIXES: &[&str] = &[
        "linux-vdso.so",
        "ld-linux",
        "libc.so",
        "libm.so",
        "libpthread.so",
        "libdl.so",
        "librt.so",
        "libgcc_s.so",
    ];

    fn allowed(soname: &str) -> bool {
        ALLOWED_PREFIXES
            .iter()
            .any(|prefix| soname.starts_with(prefix))
    }

    #[test]
    fn debug_binary_has_only_expected_dynamic_libs() {
        let bin = cargo_bin("syllabix");
        let output = Command::new("ldd")
            .arg(&bin)
            .output()
            .expect("ldd must be available on linux CI");
        assert!(
            output.status.success(),
            "ldd failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("statically linked") {
            return;
        }

        let mut unexpected = Vec::new();
        for line in stdout.lines() {
            let soname = line.split_whitespace().next().unwrap_or("");
            let soname = soname.rsplit('/').next().unwrap_or(soname);
            if soname.is_empty() {
                continue;
            }
            if !allowed(soname) {
                unexpected.push(soname.to_string());
            }
        }

        assert!(
            unexpected.is_empty(),
            "undeclared dynamic dependencies {unexpected:?}\n{stdout}"
        );
    }
}
