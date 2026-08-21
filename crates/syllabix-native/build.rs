//! Build ggml once, then the whisper.cpp and llama.cpp frontends against it.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/shim.c");
    println!("cargo:rerun-if-changed=include/syllabix_native.h");
    println!("cargo:rerun-if-changed=../../vendor/llama.cpp");
    println!("cargo:rerun-if-changed=../../vendor/whisper.cpp");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    let target = env::var("TARGET").unwrap_or_default();
    let apple = target.contains("apple");
    let apple_arm = apple && (target.starts_with("aarch64") || target.contains("arm64"));

    let mut config = cmake::Config::new(".");
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_CUDA", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_OPENSSL", "OFF");

    if apple {
        config
            .define("GGML_METAL", "ON")
            .define("GGML_METAL_EMBED_LIBRARY", "ON")
            .define("GGML_BLAS", "ON")
            .define("GGML_BLAS_VENDOR", "Apple")
            .define("GGML_ACCELERATE", "ON");
        if apple_arm {
            config.define("GGML_NATIVE", "ON");
        } else {
            config.define("GGML_NATIVE", "OFF");
        }
        config.define("GGML_CPU_KLEIDIAI", "OFF");
        if let Ok(macosx) = env::var("MACOSX_DEPLOYMENT_TARGET") {
            config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &macosx);
        }
    } else {
        config
            .define("GGML_NATIVE", "OFF")
            .define("GGML_METAL", "OFF")
            .define("GGML_BLAS", "OFF")
            .define("GGML_ACCELERATE", "OFF")
            .define("GGML_OPENMP", "OFF")
            .define("GGML_CPU_KLEIDIAI", "OFF");
    }

    let dst = config.build();

    let mut search = vec![dst.join("lib"), dst.join("lib64"), dst.join("build")];
    collect_lib_dirs(&dst, &mut search);
    for dir in &search {
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    let mut libs = collect_static_libs(&search);
    // Link order: shim, frontends, then ggml pieces.
    prefer_first(&mut libs, &["syllabix_native", "whisper", "llama"]);
    // Release `--gc-sections` otherwise drops unused ggml/whisper objects
    // because the CLI does not transcribe yet. Keep both frontends in the binary.
    for lib in &libs {
        println!("cargo:rustc-link-lib=static:+whole-archive={lib}");
    }
    if apple {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=dylib=advapi32");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}

fn collect_lib_dirs(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        if name == "lib" || name == "lib64" {
            out.push(entry.path().to_path_buf());
        }
    }
}

fn collect_static_libs(dirs: &[PathBuf]) -> Vec<String> {
    let mut names = Vec::new();
    for dir in dirs {
        let Ok(read) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in read.filter_map(Result::ok) {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            let name = if ext == "a" && stem.starts_with("lib") {
                stem.trim_start_matches("lib").to_string()
            } else if ext == "lib" {
                stem.to_string()
            } else {
                continue;
            };
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

fn prefer_first(libs: &mut Vec<String>, first: &[&str]) {
    let mut ordered = Vec::new();
    for name in first {
        if let Some(index) = libs.iter().position(|lib| lib == name) {
            ordered.push(libs.remove(index));
        }
    }
    ordered.append(libs);
    *libs = ordered;
}
