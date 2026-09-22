"""AI Hub profiling harness for Snap — Snapdragon Edition.

Goal: get REAL on-device latency numbers for the quantized models on
Snapdragon X Elite / X2 Elite (the chips in Snapdragon-powered HP OmniBooks),
without owning the laptop — Qualcomm hosts the devices in the AI Hub cloud.

Setup:
    uv sync --extra aihub        # installs qai-hub
    # API token from https://workbench.aihub.qualcomm.com -> Account -> Settings
    qai-hub configure --api_token <YOUR_TOKEN>

Usage:
    # 1) See which Snapdragon X devices are available as profiling targets:
    python scripts/profile_on_aihub.py --list-devices

    # 2) Profile an exported ONNX model (e.g. a Whisper encoder export):
    python scripts/profile_on_aihub.py --onnx whisper_encoder.onnx --device "X Elite"

    # 3) For catalog LLMs (Qwen3-4B-Instruct-2507, ...), use the per-model export CLI
    #    shown on each model page at https://aihub.qualcomm.com/models — it compiles,
    #    quantizes and profiles in one command, e.g.:
    #    python -m qai_hub_models.models.<model_id>.export --device "<device from step 1>"

Save the printed tables + job JSON into docs/evidence/ — that is the benchmark
evidence the deck and docs/benchmarks.md cite.
"""

import argparse
import sys


def get_x_series_devices():
    import qai_hub as hub

    devices = hub.get_devices()
    # Match laptop-class targets: Snapdragon X Elite, X Plus, X2 Elite / X2 Plus
    keep = [
        d
        for d in devices
        if "X2 Elite" in d.name
        or "X2 Plus" in d.name
        or "X Elite" in d.name
        or "X Plus" in d.name
    ]
    return keep


def list_devices() -> None:
    devices = get_x_series_devices()
    if not devices:
        print(
            "No Snapdragon X devices found. Run `qai-hub configure` first, or check "
            "https://aihub.qualcomm.com for the current device list."
        )
        return
    print("Available Snapdragon X-series profiling targets:\n")
    for d in devices:
        attrs = getattr(d, "attributes", []) or []
        print(f"  - {d.name}" + (f"   [{', '.join(attrs)}]" if attrs else ""))


def profile_onnx(onnx_path: str, device_query: str) -> None:
    import qai_hub as hub

    devices = get_x_series_devices()
    matches = [d for d in devices if device_query.lower() in d.name.lower()]
    if not matches:
        print(f"No device matching '{device_query}'. Available: {[d.name for d in devices]}")
        sys.exit(1)
    device = matches[0]
    print(f"Profiling {onnx_path} on {device.name} ...")

    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        options="--target_runtime onnx",  # ONNX Runtime + QNN EP path for WoS laptops
    )
    compile_model = compile_job.get_target_model()
    print(f"Compile job {compile_job.job_id} done.")

    profile_job = hub.submit_profile_job(model=compile_model, device=device)
    prof = profile_job.download_profile()
    print(f"\n## Profile — {onnx_path} @ {device.name}\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Estimated inference time | {prof['estimated_inference_time'] / 1000:.2f} ms |")
    print(
        f"| Peak memory (estimated) | {prof.get('estimated_peak_memory_range', {}).get('primary_size', 'n/a')} B |"
    )
    cu = prof.get("compute_unit", {})
    print(f"| Compute unit | {cu} |")
    print("\nFull JSON:", prof)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--onnx", help="Path to exported ONNX model")
    ap.add_argument("--device", default="X Elite", help="Substring of target device name")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
    elif args.onnx:
        profile_onnx(args.onnx, args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
