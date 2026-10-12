# Reverse Engineering a Multi-Stage Implant Targeting Vietnamese Organizations

This repository contains the source code, detection rules, and analysis tooling accompanying the paper:

> **"Reverse Engineering a Multi-Stage Implant Targeting Vietnamese Organizations"**
> Luong Minh Anh
> Submitted to [Virus Bulletin 2026](https://www.virusbulletin.com/)

The paper presents the reverse engineering of a multi-stage implant framework discovered during incident response engagements involving targeted intrusions against organizations in Vietnam, conducted between August 2025 and February 2026. The framework features DLL sideloading, shellcode execution, service persistence, image-based payload delivery, a persistent cross-stage API proxy, a sidecar transport proxy (designated xReflect), and a C2 beacon with 10 built-in commands.

---

## Research Goals

1. Reconstruct the framework's nine-stage execution model and its single-process architecture.
2. Analyze two distinct obfuscation families (Family A: junk insertion; Family B: control flow flattening with opaque predicates) and develop automated deobfuscation tools for each.
3. Document the custom CRC64 cryptographic primitive that serves seven roles across four independently-obfuscated memory regions, forming the strongest single-authorship fingerprint.
4. Reverse engineer the persistent cross-stage API proxy mechanism, including both the event-synchronized primary wrapper and the direct call-stack spoofing auxiliary wrapper.
5. Reconstruct the image-based payload delivery pipeline using Windows Imaging Component (WIC).
6. Recover MSVC x64 RTTI from a process memory dump to reconstruct 41 C++ classes in the xReflect transport module.
7. Provide actionable detection guidance (YARA rules, Suricata rules, behavioral indicators).

---

## Repository Structure

```
.
├── deobfuscation/                        # Binary Ninja plugins for automated deobfuscation
│   ├── stage1_global_variable_junk.py    # Family A, Technique 1: global variable junk removal
│   ├── stage1_junk_array_and_null_variable.py  # Family A, Technique 2: junk array + null variable removal
│   ├── stage2_junk_code.py               # Family A applied to Stage 2 (lower threshold variant)
│   ├── stage3_control_flow_flattening.py # Family B, Phase 1: CFF reconstruction (cmov-to-jcc)
│   ├── stage3_opaque_predicate.py        # Family B, Phase 2: opaque predicate removal
│   └── stage3_opaque_predicate_test.py   # Test and debug utilities for opaque predicate removal
│
├── decryption/                           # Standalone decryption tools
│   ├── decrypt_config.py                 # Configuration blob decryptor (CRC64 stream cipher)
│   └── decrypt_image.py                  # Image-based payload decryptor (full type=3 pipeline)
│
├── RTTI/                                 # RTTI recovery tooling
│   └── rtti_recover.py                   # MSVC x64 RTTI bootstrap from memory dump (Binary Ninja)
│
├── detection/                            # Network and host-based detection rules
│   ├── yara/
│   │   └── rule.yar                      # YARA rule: SetLastError dispatcher pattern
│   └── suricata/
│       └── detect_tls_cert.rules         # Suricata rules: xReflect fake DigiCert certificate
│
└── README.md
```

### Relationship to Paper Sections

| Tool | Paper Section | Purpose |
|------|---------------|---------|
| `stage1_global_variable_junk.py` | Section 3.2 (Family A, Technique 1) | Remove junk code referencing high-refcount global variables |
| `stage1_junk_array_and_null_variable.py` | Section 3.2 (Family A, Technique 2) | Remove junk array taint and null-variable identity operations |
| `stage2_junk_code.py` | Section 3.2 (Family A) | Same algorithm as Stage 1 global variable junk, adapted for Stage 2 density |
| `stage3_control_flow_flattening.py` | Section 3.3 (Family B, Phase 1) | Reconstruct original CFG from CFF dispatcher; convert cmov to jcc |
| `stage3_opaque_predicate.py` | Section 3.3 (Family B, Phase 2) | Identify paired state variables and patch jne to unconditional jmp |
| `stage3_opaque_predicate_test.py` | Section 3.3 (Family B) | Interactive test utilities for opaque predicate deobfuscation |
| `decrypt_config.py` | Section 5.1 (CRC64 Stream Cipher) | Decrypt configuration blobs using reconstructed CRC64 cipher |
| `decrypt_image.py` | Section 6 (Image-Based Payload Delivery) | Full image decryption pipeline: WIC decode, key derivation, CRC64 decrypt |
| `rtti_recover.py` | Section 8 (RTTI Recovery) | Recover MSVC x64 RTTI from headerless memory dumps via pSelf bootstrap |
| `rule.yar` | Section 10.1 (Behavioral Detection) | Detect SetLastError dispatcher pattern in x64 PE binaries |
| `detect_tls_cert.rules` | Section 10.2 (Network Indicators) | Detect fraudulent DigiCert TLS certificate with characteristic typos |

---

## Dependencies

### Deobfuscation Tools (Binary Ninja Plugins)

- [Binary Ninja](https://binary.ninja/) (tested with version 4.x)
- [Triton](https://triton-library.github.io/) dynamic symbolic execution framework (required by `stage1_*.py` and `stage2_junk_code.py`)
- [Capstone](https://www.capstone-engine.org/) disassembly framework (required by `stage3_opaque_predicate*.py`)

### Decryption Tools (Standalone Python)

- Python 3.10+
- [Pillow](https://pillow.readthedocs.io/) (required by `decrypt_image.py` for image decoding)
- No external dependencies for `decrypt_config.py` (uses only the Python standard library)

### RTTI Recovery

- [Binary Ninja](https://binary.ninja/) (tested with version 4.x)
- No additional Python packages required

### Detection Rules

- [YARA](https://yara.readthedocs.io/) 4.x+ for `rule.yar`
- [Suricata](https://suricata.io/) 6.x+ for `detect_tls_cert.rules`

---

## Build and Installation

No build step is required. All tools are Python scripts executed either standalone or within Binary Ninja's script console.

To install Python dependencies for the standalone decryption tools:

```bash
pip install pillow
```

To install Triton for the deobfuscation plugins, follow the [Triton installation guide](https://triton-library.github.io/documentation/doxygen/index.html#install_sec). Triton must be importable from Binary Ninja's Python environment.

---

## Usage

### Deobfuscation: Family A (Stages 1-2)

The Family A deobfuscators are Binary Ninja plugins. Execute them from the Binary Ninja Script Console (`Ctrl+``):

**Technique 1 -- Global variable junk removal** (Section 3.2):

```python
# Automatically identifies data variables with >= 100 code references,
# backward-traces register dependencies via Triton, and NOP-patches junk.
# Execute in Binary Ninja Script Console with the target binary loaded.
exec(open("deobfuscation/stage1_global_variable_junk.py").read())
```

**Technique 2 -- Junk array and null variable removal** (Section 3.2):

```python
from deobfuscation.stage1_junk_array_and_null_variable import JunkArrayDeobfuscator

# Single function
d = JunkArrayDeobfuscator(bv, bv.get_function_at(0xTARGET_ADDR))
d.run()     # Highlight junk (red = tainted, orange = null no-op)
d.patch()   # NOP-patch highlighted instructions

# All functions
for func in bv.functions:
    JunkArrayDeobfuscator(bv, func).run()
```

### Deobfuscation: Family B (Stage 3+)

Family B deobfuscation is a two-phase process. CFF must be resolved first.

**Phase 1 -- Control flow flattening reconstruction** (Section 3.3):

```python
from deobfuscation.stage3_control_flow_flattening import deobfuscate_cff

# Single function
deobfuscate_cff(bv, bv.get_function_at(0xTARGET_ADDR))

# All functions (batch mode)
exec(open("deobfuscation/stage3_control_flow_flattening.py").read())
```

**Phase 2 -- Opaque predicate removal** (Section 3.3):

```python
# Scans for paired DWORD state variables, resolves always-taken branches,
# and patches conditional JNE to unconditional JMP.
exec(open("deobfuscation/stage3_opaque_predicate.py").read())
```

### Decryption: Configuration Blob

```bash
python decryption/decrypt_config.py <encrypted_config> <output_file>
```

The script expects an encrypted configuration file where the first 8 bytes are the random seed (nonce) and the remainder is ciphertext. The hardcoded passphrase (`0x6b92d1638c205f10`) is specific to the analyzed samples.

### Decryption: Image-Based Payload

```bash
python decryption/decrypt_image.py <carrier_image.png> -o decrypted_payload.bin
```

This reconstructs the full type=3 decryption pipeline described in Section 6: WIC-equivalent image decoding (via Pillow), extraction of the 8-byte seed from the pixel buffer header, CRC64 key derivation using the hardcoded passphrase (`0x06C28137D91D2C1F0`), and stream cipher decryption of the remaining pixel data.

### RTTI Recovery

```python
# In Binary Ninja Script Console with the memory dump loaded:
# 1. Adjust IMAGE_BASE, CODE_START/END, DATA_START/END, TYPE_INFO_VFTABLE
#    constants at the top of the script to match your dump.
# 2. Set DRY_RUN = True for preview, False to apply renames.
exec(open("RTTI/rtti_recover.py").read())
```

The script performs three scan passes over a single `MemoryBuffer` to recover `CompleteObjectLocator` structures via the `pSelf` bootstrap technique (Section 8.2), reconstruct vftables, and resolve class inheritance hierarchies.

### Detection Rules

**YARA** (Section 10.1):

```bash
yara detection/yara/rule.yar <suspect_binary>
```

Detects the `SetLastError` dispatcher pattern: repeated `lea rcx, [rip+imm32]` followed by `call qword ptr [rip+imm32]` sequences within a single function.

**Suricata** (Section 10.2):

```bash
# Add to your Suricata rules directory and reload:
cp detection/suricata/detect_tls_cert.rules /etc/suricata/rules/
suricata-update
```

Detects the xReflect proxy's fraudulent TLS certificate containing the typos "San Diago" (correct: "San Diego") and "digi-cert.com" (correct: "digicert.com").

---

## Sample Data and Reproducibility

This repository contains **analysis tools only**. No malware samples, memory dumps, encrypted configuration files, or carrier images are included.

To reproduce the results described in the paper:

1. **Deobfuscation tools** require the original obfuscated binaries loaded in Binary Ninja. The YARA rule (`rule.yar`) can be used to identify candidate samples. IoCs in Appendix B of the paper provide additional sample-identification criteria.
2. **Decryption tools** require the encrypted configuration files or carrier images extracted during analysis. The hardcoded passphrases and CRC64 table are embedded in the scripts.
3. **RTTI recovery** requires a process memory dump of the xReflect module. The script's address constants (`IMAGE_BASE`, `CODE_START`, `DATA_START`, `TYPE_INFO_VFTABLE`) must be adjusted to match the dump's memory layout.

Researchers with access to the analyzed samples can use the IoCs provided in the paper (Appendix B) to locate them via malware repositories or threat intelligence platforms.

---

## Limitations

- **Binary Ninja dependency.** The deobfuscation tools and RTTI recovery script are implemented as Binary Ninja plugins and are not portable to other disassemblers without modification.
- **Sample-specific constants.** The decryption scripts contain hardcoded passphrases, XOR constants, and CRC64 tables extracted from the analyzed samples. Applying these tools to different samples or variants may require updating these constants.
- **Address-specific configuration.** The RTTI recovery script requires manual adjustment of memory region boundaries and the `type_info::vftable` address for each new memory dump.
- **Single-variant validation.** The deobfuscation tools were developed and validated against the three variants described in the paper (RNP, Zoom, MEGAsync). Their effectiveness on other samples using similar but not identical obfuscation is not guaranteed.
- **No dynamic analysis.** The decryption tools reconstruct the cipher offline. They do not replicate the runtime environment checks (e.g., regex-based analyst-environment detection) described in Section 6.2.
- **Image decoding approximation.** The image decryptor uses Pillow to approximate Windows Imaging Component (WIC) decoding. Minor pixel-level differences between Pillow and WIC decoders may affect decryption of edge-case image formats.

---

## Citation

If you use these tools or reference this work, please cite:

```bibtex
@inproceedings{luong2026multistage,
  title     = {Reverse Engineering a Multi-Stage Implant Targeting
               {Vietnamese} Organizations},
  author    = {Luong, Minh Anh},
  booktitle = {Virus Bulletin Conference (VB2026)},
  year      = {2026},
  note      = {To appear}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Disclaimer

The tools in this repository are provided for **defensive security research and incident response purposes only**. They are designed to aid in the analysis of malicious software and the development of detection capabilities. The authors do not condone the use of these tools for unauthorized access to computer systems or any other illegal activity.
