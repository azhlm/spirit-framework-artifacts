#!/usr/bin/env python3

import struct

# ====================================================================
# CONFIGURATION — Điều chỉnh cho binary của bạn
# ====================================================================

# Image base (tính từ COL.pSelf, đã verify 2 lần)
IMAGE_BASE = 0x277a6650C30

# Code segment (RX) — validate vftable entries phải trỏ vào đây
CODE_START = 0x277a7160000
CODE_END   = 0x277a7415000

# Data segment (RW) — chứa RTTI metadata, vftables, strings
# Script sẽ đọc toàn bộ vùng này vào RAM một lần
DATA_START = 0x277a6550000
DATA_END   = 0x277a6878000

# type_info::vftable — giá trị pVFTable trong mọi TypeDescriptor
# (8 bytes đầu mỗi TypeDescriptor đều giống nhau)
TYPE_INFO_VFTABLE = 0x277a67a13e0

# Nếu True: chỉ in kết quả, không rename
DRY_RUN = True

# ====================================================================
# HELPER: Đọc data segment một lần
# ====================================================================

class MemoryBuffer:
    """
    Đọc toàn bộ data segment vào RAM để search nhanh.
    Thay vì gọi bv.read() hàng trăm lần qua API (chậm),
    ta đọc 1 lần rồi dùng Python bytes operations (nhanh).
    """
    def __init__(self, bv, start, end):
        self.start = start
        self.end = end
        self.size = end - start
        print(f"  Reading memory 0x{start:X}..0x{end:X} ({self.size} bytes / {self.size/1024/1024:.1f} MB)")
        self.data = bv.read(start, self.size)
        if len(self.data) != self.size:
            print(f"  WARNING: requested {self.size} bytes, got {len(self.data)}")
            self.size = len(self.data)
        print(f"  Done — {self.size} bytes in memory")
    
    def read(self, addr, length):
        """Đọc bytes tại absolute address."""
        off = addr - self.start
        if off < 0 or off + length > self.size:
            return b''
        return self.data[off:off+length]
    
    def read_u32(self, addr):
        d = self.read(addr, 4)
        return struct.unpack('<I', d)[0] if len(d) == 4 else None
    
    def read_u64(self, addr):
        d = self.read(addr, 8)
        return struct.unpack('<Q', d)[0] if len(d) == 8 else None
    
    def find_all(self, pattern, align=1):
        """
        Tìm tất cả vị trí của pattern trong buffer.
        Returns: list of absolute addresses.
        """
        results = []
        pos = 0
        while True:
            idx = self.data.find(pattern, pos)
            if idx < 0:
                break
            abs_addr = self.start + idx
            if align == 1 or (idx % align == 0):
                results.append(abs_addr)
            pos = idx + 1
        return results


# ====================================================================
# PHASE 1: Thu thập TypeDescriptor
# ====================================================================

def find_type_descriptors(bv):
    """
    Scan string table tìm ".?AV" prefix.
    Lùi 0x10 bytes → TypeDescriptor address.
    Validate pVFTable == TYPE_INFO_VFTABLE.
    
    Returns: dict {td_address: class_name}
    """
    type_descs = {}
    
    for s in bv.get_strings():
        val = s.value
        if not val.startswith(".?AV"):
            continue
        
        # ".?AVCClient@@" → "CClient"
        name = val[4:].rstrip('@')
        
        # TypeDescriptor = string_addr - 0x10
        td_addr = s.start - 0x10
        
        pvft_bytes = bv.read(td_addr, 8)
        if len(pvft_bytes) < 8:
            continue
        pvft = struct.unpack('<Q', pvft_bytes)[0]
        
        if pvft == TYPE_INFO_VFTABLE:
            type_descs[td_addr] = name
    
    return type_descs


# ====================================================================
# PHASE 2: Tìm COL → vftable → inheritance (in-memory)
# ====================================================================

def find_all_cols(mem):
    """
    Scan data segment tìm TẤT CẢ COL structures một lần.
    
    COL signature: DWORD=1 tại offset +0x00, và validate
    IMAGE_BASE + pSelf == candidate_address.
    
    Trả về dict: {col_addr: parsed_col_dict}
    
    Đây là bước tối ưu chính: thay vì scan N lần cho N classes,
    ta scan 1 lần tìm tất cả COL, rồi lookup theo td_rva.
    """
    cols = {}
    
    # Scan từng DWORD-aligned position
    sig_pattern = struct.pack('<I', 1)  # signature = 1
    
    for addr in mem.find_all(sig_pattern, align=4):
        # Đọc COL candidate (24 bytes)
        col_data = mem.read(addr, 24)
        if len(col_data) < 24:
            continue
        
        sig, offset, cd_off, p_td, p_ch, p_self = struct.unpack('<IIIIII', col_data)
        
        # sig đã match (=1), giờ validate pSelf
        if IMAGE_BASE + p_self != addr:
            continue
        
        # Thêm sanity check: pTypeDescriptor và pClassHierarchy 
        # phải trỏ vào data segment
        td_abs = IMAGE_BASE + p_td
        ch_abs = IMAGE_BASE + p_ch
        if not (mem.start <= td_abs < mem.end and mem.start <= ch_abs < mem.end):
            continue
        
        cols[addr] = {
            'col_addr': addr,
            'offset': offset,
            'cd_offset': cd_off,
            'td_rva': p_td,
            'ch_rva': p_ch,
            'self_rva': p_self,
            'td_abs': td_abs,
            'ch_abs': ch_abs,
        }
    
    return cols


def build_td_rva_to_col_map(cols):
    """
    Build reverse lookup: td_rva → [list of COLs].
    
    Một class có thể có nhiều COL nếu có multiple inheritance
    (mỗi base class interface có vftable riêng, offset ≠ 0).
    """
    td_to_cols = {}
    for col in cols.values():
        rva = col['td_rva']
        if rva not in td_to_cols:
            td_to_cols[rva] = []
        td_to_cols[rva].append(col)
    return td_to_cols


def find_vftables_for_all_cols(mem, cols):
    """
    Tìm vftable cho TẤT CẢ COL trong một pass scan.
    
    Nguyên lý: vftable[-1] (QWORD ngay trước vftable) = absolute ptr đến COL.
    Scan data tìm QWORD == col_addr, check QWORD tiếp theo là code pointer.
    
    Tối ưu: build set of target addresses, scan data MỘT LẦN,
    check mỗi QWORD có match target nào không.
    """
    # Build lookup: col_addr → col_dict
    col_addrs = set(cols.keys())
    
    # Scan toàn bộ data segment, kiểm tra mỗi QWORD
    results = {}  # col_addr → (vftable_addr, [vfunc_addrs])
    
    for i in range(0, mem.size - 16, 8):  # QWORD-aligned
        qword = struct.unpack_from('<Q', mem.data, i)[0]
        
        if qword not in col_addrs:
            continue
        
        # Found: mem[i] = pointer đến COL
        # → vftable bắt đầu tại i+8
        ptr_addr = mem.start + i
        vft_addr = ptr_addr + 8
        
        # Đếm vftable entries (liên tiếp code pointers)
        vfuncs = []
        j = i + 8
        while j + 8 <= mem.size:
            entry = struct.unpack_from('<Q', mem.data, j)[0]
            if CODE_START <= entry < CODE_END:
                vfuncs.append(entry)
                j += 8
            else:
                break
        
        if vfuncs:
            col_addr = qword
            # Có thể có nhiều references đến cùng COL (hiếm)
            # Chỉ lấy cái đầu tiên có valid vfuncs
            if col_addr not in results:
                results[col_addr] = (vft_addr, vfuncs)
    
    return results


def get_inheritance(mem, ch_rva):
    """
    Parse ClassHierarchyDescriptor → BaseClassArray → tên base classes.
    
    Returns: list of class names [self, parent, grandparent, ...]
    """
    chd_addr = IMAGE_BASE + ch_rva
    chd_data = mem.read(chd_addr, 16)
    if len(chd_data) < 16:
        return []
    
    _, attributes, num_bases, bca_rva = struct.unpack('<IIII', chd_data)
    
    if num_bases <= 0 or num_bases > 20:
        return []
    
    bca_addr = IMAGE_BASE + bca_rva
    bca_data = mem.read(bca_addr, num_bases * 4)
    if len(bca_data) < num_bases * 4:
        return []
    
    names = []
    for i in range(num_bases):
        bcd_rva = struct.unpack_from('<I', bca_data, i * 4)[0]
        bcd_addr = IMAGE_BASE + bcd_rva
        
        bcd_data = mem.read(bcd_addr, 4)
        if len(bcd_data) < 4:
            names.append("???")
            continue
        
        base_td_rva = struct.unpack('<I', bcd_data)[0]
        base_td_addr = IMAGE_BASE + base_td_rva
        
        # Đọc tên từ TypeDescriptor.name (+0x10)
        name_data = mem.read(base_td_addr + 0x10, 64)
        if not name_data:
            names.append("???")
            continue
        
        null_idx = name_data.find(b'\x00')
        if null_idx > 0:
            raw = name_data[:null_idx].decode('ascii', errors='replace')
            if raw.startswith('.?AV'):
                names.append(raw[4:].rstrip('@'))
            else:
                names.append(raw)
        else:
            names.append("???")
    
    return names


# ====================================================================
# PHASE 3: Rename
# ====================================================================

def apply_renames(bv, results):
    """Rename vftable, COL, TypeDescriptor, và virtual functions."""
    count = 0
    
    for r in results:
        cls = r['class']
        
        # ── Rename TypeDescriptor ──
        if r['td_addr']:
            name = f"{cls}__TypeDescriptor"
            if DRY_RUN:
                print(f"  [DRY] 0x{r['td_addr']:X} → {name}")
            else:
                try:
                    bv.define_user_data_var(r['td_addr'], bv.parse_type_string("void*")[0])
                    dv = bv.get_data_var_at(r['td_addr'])
                    if dv:
                        dv.name = name
                        count += 1
                except:
                    pass
        
        # ── Rename COL ──
        if r['col_addr']:
            suffix = "" if r.get('col_offset', 0) == 0 else f"__off{r['col_offset']}"
            name = f"{cls}__RTTI_COL{suffix}"
            if DRY_RUN:
                print(f"  [DRY] 0x{r['col_addr']:X} → {name}")
            else:
                try:
                    bv.define_user_data_var(r['col_addr'], bv.parse_type_string("void*")[0])
                    dv = bv.get_data_var_at(r['col_addr'])
                    if dv:
                        dv.name = name
                        count += 1
                except:
                    pass
        
        # ── Rename vftable ──
        if r['vftable']:
            suffix = "" if r.get('col_offset', 0) == 0 else f"__off{r['col_offset']}"
            name = f"{cls}__vftable{suffix}"
            if DRY_RUN:
                print(f"  [DRY] 0x{r['vftable']:X} → {name}")
            else:
                try:
                    bv.define_user_data_var(r['vftable'], bv.parse_type_string("void*")[0])
                    dv = bv.get_data_var_at(r['vftable'])
                    if dv:
                        dv.name = name
                        count += 1
                except:
                    pass
        
        # ── Rename virtual functions ──
        # Chỉ rename nếu tên hiện tại là auto-generated (sub_xxx)
        if r['vfuncs'] and not DRY_RUN:
            for i, vf_addr in enumerate(r['vfuncs']):
                func = bv.get_function_at(vf_addr)
                if func and func.name.startswith('sub_'):
                    func.name = f"{cls}__vfunc_{i}"
                    count += 1
    
    return count


# ====================================================================
# MAIN
# ====================================================================

def run(bv):
    print("=" * 100)
    print("MSVC x64 RTTI RECOVERY — v2 (optimized)")
    print("=" * 100)
    
    # ── Phase 1 ──
    print("\n[Phase 1] Scanning TypeDescriptors...")
    type_descs = find_type_descriptors(bv)
    
    # Phân loại
    STD_KEYWORDS = ['std@@', 'type_info', 'bad_alloc', 'exception@@',
                    'logic_error', 'length_error', 'out_of_range', 'error_category']
    
    custom_tds = {a: n for a, n in type_descs.items()
                  if not any(kw in type_descs[a] or kw in n for kw in STD_KEYWORDS)}
    
    print(f"  Total: {len(type_descs)}  (custom: {len(custom_tds)}, std: {len(type_descs)-len(custom_tds)})")
    
    # ── Đọc data segment vào RAM ──
    print("\n[Loading] Reading data segment into memory...")
    mem = MemoryBuffer(bv, DATA_START, DATA_END)
    
    # ── Phase 2a: Tìm tất cả COL ──
    print("\n[Phase 2a] Scanning for CompleteObjectLocators...")
    all_cols = find_all_cols(mem)
    print(f"  Found {len(all_cols)} valid COLs")
    
    # Build td_rva → COL mapping
    td_to_cols = build_td_rva_to_col_map(all_cols)
    
    # ── Phase 2b: Tìm tất cả vftable ──
    print("\n[Phase 2b] Scanning for vftable references...")
    col_to_vft = find_vftables_for_all_cols(mem, all_cols)
    print(f"  Found {len(col_to_vft)} vftables")
    
    # ── Phase 2c: Tổng hợp ──
    print("\n[Phase 2c] Building class hierarchy...")
    
    results = []
    
    for td_addr, class_name in sorted(custom_tds.items(), key=lambda x: x[1]):
        td_rva = td_addr - IMAGE_BASE
        
        if td_rva not in td_to_cols:
            results.append({
                'class': class_name, 'td_addr': td_addr,
                'col_addr': None, 'col_offset': 0,
                'vftable': None, 'vfuncs': [], 'bases': [],
            })
            continue
        
        for col in td_to_cols[td_rva]:
            col_addr = col['col_addr']
            
            # Vftable
            vft_addr, vfuncs = col_to_vft.get(col_addr, (None, []))
            
            # Inheritance
            bases = get_inheritance(mem, col['ch_rva'])
            base_names = bases[1:] if len(bases) > 1 else []
            
            results.append({
                'class': class_name, 'td_addr': td_addr,
                'col_addr': col_addr, 'col_offset': col['offset'],
                'vftable': vft_addr, 'vfuncs': vfuncs,
                'bases': base_names,
            })
    
    # ── In kết quả ──
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"\n  {'Class':<30} {'vftable':<20} {'#vf':>3}  {'Inherits from'}")
    print("  " + "-" * 90)
    
    for r in sorted(results, key=lambda x: x['class']):
        vft = f"0x{r['vftable']:X}" if r['vftable'] else "NOT FOUND"
        base = " → ".join(r['bases']) if r['bases'] else "(root)"
        off = f" [off={r['col_offset']}]" if r.get('col_offset', 0) != 0 else ""
        print(f"  {r['class']:<30} {vft:<20} {len(r['vfuncs']):>3}  {base}{off}")
    
    # ── Inheritance tree ──
    print("\n" + "=" * 100)
    print("INHERITANCE TREE")
    print("=" * 100)
    
    # Build parent map (chỉ primary vftable, offset=0)
    parent_map = {}
    for r in results:
        if r.get('col_offset', 0) != 0:
            continue
        parent_map[r['class']] = r['bases'][0] if r['bases'] else None
    
    def print_tree(parent, depth=1):
        children = sorted([c for c, p in parent_map.items() if p == parent])
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            r = next((x for x in results if x['class'] == child and x.get('col_offset',0)==0), None)
            nv = len(r['vfuncs']) if r else 0
            connector = "└── " if is_last else "├── "
            print(f"  {'│   ' * (depth-1)}{connector}{child} ({nv} vfuncs)")
            print_tree(child, depth + 1)
    
    roots = set()
    for cls, parent in parent_map.items():
        if parent is None:
            roots.add(cls)
        elif parent not in parent_map:
            roots.add(parent)
    
    for root in sorted(roots):
        r = next((x for x in results if x['class'] == root and x.get('col_offset',0)==0), None)
        nv = len(r['vfuncs']) if r else 0
        print(f"\n  {root} ({nv} vfuncs)")
        print_tree(root, 1)
    
    # ── Vftable entries ──
    print("\n" + "=" * 100)
    print("VFTABLE ENTRIES")
    print("=" * 100)
    
    for r in sorted(results, key=lambda x: x['class']):
        if not r['vfuncs'] or r.get('col_offset', 0) != 0:
            continue
        print(f"\n  {r['class']}::vftable @ 0x{r['vftable']:X}  ({len(r['vfuncs'])} entries)")
        for i, vf in enumerate(r['vfuncs']):
            func = bv.get_function_at(vf)
            fname = func.name if func else "???"
            print(f"    [{i:>2}] 0x{vf:X}  {fname}")
    
    # ── Phase 3: Rename ──
    print("\n" + "=" * 100)
    if DRY_RUN:
        print("RENAME PREVIEW (DRY_RUN=True, nothing changed)")
    else:
        print("APPLYING RENAMES")
    print("=" * 100)
    
    count = apply_renames(bv, results)
    print(f"\n  {'Would rename' if DRY_RUN else 'Renamed'} {count} labels")
    
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    
    return results


# ====================================================================
# ENTRY POINT
# ====================================================================

# Lấy BinaryView — tương thích với cả console và Run Script
try:
    # Thử lấy từ UI context (Tools > Run Script)
    import binaryninjaui
    ctx = binaryninjaui.UIContext.activeContext()
    handler = ctx.contentActionHandler()
    actionCtx = handler.actionContext()
    _bv = actionCtx.binaryView
except:
    try:
        # Thử biến global 'bv' từ Script Console
        _bv = bv  # noqa
    except NameError:
        # Thử 'current_view' 
        try:
            _bv = current_view  # noqa
        except NameError:
            print("ERROR: Không tìm thấy BinaryView")
            print("Paste script vào Script Console (Ctrl+`) rồi chạy:")
            print("  run(bv)")
            _bv = None

if _bv is not None:
    run(_bv)