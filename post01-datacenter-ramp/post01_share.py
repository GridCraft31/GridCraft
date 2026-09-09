#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 โหลด Data Center กับ ramp rate — ทำไม "กี่เมกะวัตต์" อย่างเดียวไม่พอ
================================================================================

โจทย์
    Data Center 100 MW ต่อเข้าระบบ 115 kV บนสายเส้นเดียวกับสถานีจำหน่าย
    115/22 kV ที่จ่ายชุมชนอยู่ก่อนแล้ว

    GPU หลายพันใบในคลัสเตอร์เดียวถูกบังคับให้เดินเป็นจังหวะเดียวกันด้วย
    global barrier ของ training loop จึงไม่มี diversity มากลบกัน
    โหลดทั้งก้อนขยับพร้อมกัน = +80 MW ในหนึ่งวินาที

    คำถาม: แรงดันในระบบตกเท่าไร และต่างจากกรณีที่กินไฟเท่ากัน
            แต่บังคับให้ค่อย ๆ ไต่ขึ้นแค่ไหน

วิธีทำ
    quasi-static power flow ความละเอียด 1 วินาที รวม 600 จุด (pandapower)
    ไม่ต้องใช้ EMT เพราะปรากฏการณ์ที่สนใจอยู่ที่สเกลวินาทีถึงนาที
    ซึ่งเป็นจังหวะการทำงานของ OLTC ไม่ใช่จังหวะสวิตชิ่ง

วิธีรัน
    pip install pandapower matplotlib
    python post01_share.py

    บน Google Colab
        !pip install -q pandapower
        !apt-get -qq install fonts-tlwg-garuda      # ฟอนต์ไทย แล้วรีสตาร์ต runtime
        แล้วก๊อปไฟล์นี้ทั้งไฟล์วางในเซลล์เดียว กด Run ได้เลย

ที่มาของตัวเลขที่ใช้
    [1] NERC Large Loads Task Force, "Characteristics and Risks of Emerging
        Large Loads," July 2025 — ลักษณะโหลดและอัตราการเปลี่ยนแปลงระดับ
        สิบถึงร้อย MW/s ตอน job start / checkpoint / fault recovery
        https://www.nerc.com/globalassets/who-we-are/standing-committees/rstc/
        3_doc_white-paper-characteristics-and-risks-of-emerging-large-loads.pdf
    [2] ค่า R ของตัวนำ ACSR Drake 795 kcmil — datasheet ของผู้ผลิต
    [3] ค่า X ของสายเหนือดิน 115 kV ขึ้นกับระยะห่างเฟสและรูปเสา
        ค่าที่พบทั่วไปอยู่ในช่วงราว 0.31–0.44 Ω/km
        อ้างอิงมาตรฐาน: Westinghouse, Electrical Transmission and
        Distribution Reference Book
        ★ ถ้าจะใช้งานจริง ควรคำนวณจากรูปเสาของระบบตัวเอง

ค่าที่เป็นสมมติฐาน (ปรับได้ และผลลัพธ์ไวต่อค่าเหล่านี้)
    - REG_DELAY  หน่วงเวลาก่อน OLTC ขยับ tap = 45 วินาที
      ของจริงตั้งได้ตั้งแต่หลักสิบวินาทีถึงหลายนาที ถ้าเปลี่ยนค่านี้
      จำนวนวินาทีที่แรงดันหลุดเกณฑ์จะเปลี่ยนตามทันที
    - vk% ของหม้อแปลงทั้งสองตัว = 12% (ค่าที่พบบ่อยในระดับแรงดันนี้)
    - เกณฑ์แรงดัน 0.95 pu ตามกรอบ ±5% ที่ใช้กันทั่วไป

ระบบทั้งหมดเป็นระบบทดสอบสมมติ ไม่ใช่ข้อมูลของหน่วยงานใด
ตัวอย่างนี้ทำขึ้นเพื่อการเรียนรู้ งานจริงต้องตรวจสอบโดยวิศวกรผู้มีใบอนุญาต
================================================================================
"""
import os
import matplotlib

# ---- เลือก backend ตามที่รันอยู่ ------------------------------------------
# รันเป็นสคริปต์บนเครื่องที่ไม่มีจอ ต้องใช้ "Agg"
# แต่ถ้าอยู่ใน notebook ห้ามใช้ Agg ไม่งั้นรันผ่านแต่ไม่มีรูปโผล่
try:
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None
except Exception:
    IN_NOTEBOOK = False
if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandapower as pp

OUT_PNG = "datacenter_ramp.png"

# ---- ฟอนต์ไทย -------------------------------------------------------------
# Google Colab ไม่มีฟอนต์ไทยติดมา ถ้าไม่จัดการ matplotlib จะถอยไปใช้ DejaVu Sans
# ซึ่งไม่มีอักขระไทย ผลคือตัวหนังสือกลายเป็นสี่เหลี่ยมและมี UserWarning ท่วมจอ
#
# ลำดับที่ลอง: ฟอนต์ที่ติดตั้งในเครื่องอยู่แล้ว -> ถ้าไม่มีค่อยโหลด Noto Sans Thai
# มาเก็บไว้ข้าง ๆ ไฟล์นี้ (ครั้งเดียว ไม่ต้อง apt ไม่ต้องรีสตาร์ต runtime)
#
# ต้องตั้ง font.family เป็น list เสมอ เพื่อให้ fallback ไปหา DejaVu
# สำหรับอักขระที่ฟอนต์ไทยไม่มี เช่น Ω และ →
LOCAL_FONTS = [
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/thai/Garuda.ttf",
    "NotoSansThai-Regular.ttf",
]
NOTO_URL = ("https://github.com/googlefonts/noto-fonts/raw/main/"
            "hinted/ttf/NotoSansThai/NotoSansThai-Regular.ttf")


def _use_thai_font():
    path = next((f for f in LOCAL_FONTS if os.path.exists(f)), None)
    if path is None:
        try:
            import urllib.request
            path = "NotoSansThai-Regular.ttf"
            urllib.request.urlretrieve(NOTO_URL, path)
            print("ดาวน์โหลดฟอนต์ไทย Noto Sans Thai มาใช้แล้ว")
        except Exception as e:
            print(f"เตือน: หาฟอนต์ไทยไม่ได้ ({e}) ตัวหนังสือไทยจะเป็นสี่เหลี่ยม")
            return
    font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = [
        font_manager.FontProperties(fname=path).get_name(), "DejaVu Sans"]


_use_thai_font()

# =============================================================== พารามิเตอร์
DC_IDLE, DC_FULL = 20.0, 100.0       # MW   Data Center ตอน idle / ตอนเทรนเต็มกำลัง
NEIGHBOR_P, NEIGHBOR_Q = 25.0, 8.0   # MW / MVAr  ชุมชนที่สถานีจำหน่ายจ่ายอยู่
L_SHARED, L_SPUR = 15, 10            # km   สาย 115 kV ช่วงใช้ร่วมกัน / ช่วงแยกเข้า DC
LINE_R, LINE_X = 0.074, 0.40         # Ω/km ACSR Drake 795 kcmil ที่ 115 kV  [2][3]
V_LIMIT = 0.95                       # pu   เกณฑ์แรงดันต่ำสุด
REG_DB, REG_DELAY = 0.015, 45        # deadband 1.5% · หน่วง 45 วินาที (สมมติฐาน)
TAP_MIN, TAP_MAX, TAP_STEP = -16, 16, 0.625     # ช่วง tap และขนาดขั้น (%)
T_END = 600                          # วินาที
RAMP_LIMIT = 0.5                     # MW/s  อัตราที่บังคับในกรณีเปรียบเทียบ


# ================================================== 1) สร้างระบบไฟฟ้า
def build_grid():
    """ระบบส่ง 230 kV → 230/115 kV → สาย 115 kV → สถานีจำหน่าย 115/22 kV → ชุมชน
                                                └→ สาย 115 kV → Data Center

    จุดที่ต้องระวังตอนวางผัง
      * Data Center ขนาดร้อย MW ต่อที่ 115 kV ไม่ใช่ 22 kV
        (สายจำหน่าย 22 kV ทั้งเส้นรับได้ราวสิบ MW เท่านั้น)
      * สถานีจำหน่ายต้องแยกออกจาก "สายช่วงที่ใช้ร่วมกัน" ไม่ใช่อยู่ที่บัสต้นทาง
        ไม่งั้นกระแสของ DC จะไม่ไหลผ่านอิมพีแดนซ์ที่แชร์กัน แล้วจะไม่เห็นผลอะไรเลย
    """
    net = pp.create_empty_network()
    b230 = pp.create_bus(net, vn_kv=230.0, name="ระบบส่ง 230 kV")
    b115 = pp.create_bus(net, vn_kv=115.0, name="บัส 115 kV ต้นทาง")
    bsub = pp.create_bus(net, vn_kv=115.0, name="สถานีจำหน่าย 115 kV")
    bdc = pp.create_bus(net, vn_kv=115.0, name="จุดต่อ Data Center")
    b22 = pp.create_bus(net, vn_kv=22.0, name="บัส 22 kV")

    pp.create_ext_grid(net, b230, vm_pu=1.0)
    pp.create_transformer_from_parameters(
        net, b230, b115, sn_mva=200, vn_hv_kv=230, vn_lv_kv=115,
        vkr_percent=0.3, vk_percent=12, pfe_kw=0, i0_percent=0)
    for a, b, km in ((b115, bsub, L_SHARED), (bsub, bdc, L_SPUR)):
        pp.create_line_from_parameters(
            net, a, b, length_km=km, r_ohm_per_km=LINE_R, x_ohm_per_km=LINE_X,
            c_nf_per_km=0, max_i_ka=0.9)

    # หม้อแปลงสถานีจำหน่าย — ตัวนี้คือตัวที่มี OLTC คุมแรงดันฝั่ง 22 kV
    i_t2 = pp.create_transformer_from_parameters(
        net, bsub, b22, sn_mva=50, vn_hv_kv=115, vn_lv_kv=22,
        vkr_percent=0.5, vk_percent=12, pfe_kw=0, i0_percent=0,
        tap_side="hv", tap_neutral=0, tap_min=TAP_MIN, tap_max=TAP_MAX,
        tap_step_percent=TAP_STEP, tap_pos=0, tap_changer_type="Ratio")
    # หมายเหตุ pandapower 3.x: ถ้าไม่ใส่ tap_changer_type="Ratio"
    # ค่า tap_pos จะถูกเมินเงียบ ๆ power flow รันผ่านแต่แรงดันไม่ขยับเลย

    pp.create_load(net, b22, p_mw=NEIGHBOR_P, q_mvar=NEIGHBOR_Q, name="ชุมชน")
    i_dc = pp.create_load(net, bdc, p_mw=DC_IDLE, q_mvar=DC_IDLE * 0.1,
                          name="Data Center")
    return net, b22, bsub, i_dc, i_t2


# ================================================== 2) รูปโหลดสองแบบ
def profile_step():
    """โหลดจริงของ AI cluster: กระโดดทั้งก้อนเพราะ GPU เดินพร้อมกัน  [1]"""
    t = np.arange(T_END, dtype=float)
    p = np.full(T_END, DC_IDLE)
    run = slice(60, 300)
    # เทรนเต็มกำลัง + แกว่ง ±15% ที่ ~0.05 Hz ตามรอบคำนวณ gradient
    p[run] = DC_FULL + 0.15 * DC_FULL * np.sin(2 * np.pi * 0.05 * (t[run] - 60))
    p[300:330] = DC_IDLE       # checkpoint — หยุดพร้อมกันทั้งคลัสเตอร์
    p[330:] = DC_FULL
    return p


def profile_ramped(rate=RAMP_LIMIT):
    """กรณีเปรียบเทียบ: โหลดปลายทางเท่ากัน แต่บังคับให้ไต่ขึ้นช้า ๆ"""
    p = np.full(T_END, DC_IDLE)
    for k in range(60, T_END):
        p[k] = min(DC_FULL, p[k - 1] + rate)
    return p


# ================================================== 3) เดินซิม
def _settle_tap(net, b22, i_t2, limit=40):
    """ตั้ง tap เริ่มต้นให้แรงดัน 22 kV ใกล้ 1.0 เสมือน OLTC ทำงานมาก่อนหน้าแล้ว"""
    for _ in range(limit):
        pp.runpp(net, numba=False)
        v = float(net.res_bus.vm_pu.at[b22])
        if abs(v - 1.0) <= REG_DB / 2:
            break
        new = int(net.trafo.at[i_t2, "tap_pos"]) + (-1 if v < 1.0 else 1)
        if not TAP_MIN <= new <= TAP_MAX:
            break
        net.trafo.at[i_t2, "tap_pos"] = new


def simulate(p_profile):
    """เดิน power flow ทีละวินาที พร้อมตรรกะ OLTC ที่มีหน่วงเวลา

    tap อยู่ฝั่ง hv: tap_pos ติดลบ = ลดจำนวนรอบด้าน hv = ดันแรงดัน 22 kV ขึ้น
    """
    net, b22, bsub, i_dc, i_t2 = build_grid()
    _settle_tap(net, b22, i_t2)

    v22, v115, taps, off_since = [], [], [], None
    for k, p_mw in enumerate(p_profile):
        net.load.at[i_dc, "p_mw"] = float(p_mw)
        net.load.at[i_dc, "q_mvar"] = float(p_mw) * 0.1
        pp.runpp(net, numba=False)

        v = float(net.res_bus.vm_pu.at[b22])
        v22.append(v)
        v115.append(float(net.res_bus.vm_pu.at[bsub]))
        taps.append(int(net.trafo.at[i_t2, "tap_pos"]))

        # OLTC ต้องเห็นแรงดันหลุด deadband ต่อเนื่องครบ REG_DELAY ก่อนจึงขยับ 1 tap
        if abs(v - 1.0) > REG_DB:
            off_since = k if off_since is None else off_since
            if k - off_since >= REG_DELAY:
                new = int(net.trafo.at[i_t2, "tap_pos"]) + (-1 if v < 1.0 else 1)
                if TAP_MIN <= new <= TAP_MAX:
                    net.trafo.at[i_t2, "tap_pos"] = new
                off_since = None
        else:
            off_since = None
    return np.array(v22), np.array(v115), np.array(taps)


# ================================================== 4) รันทั้งสองกรณี
v22_s, v115_s, tap_s = simulate(profile_step())      # โหลดจริงของ AI cluster
v22_r, v115_r, tap_r = simulate(profile_ramped())    # บังคับ ramp ช้า
p_s, p_r = profile_step(), profile_ramped()
t = np.arange(T_END)


def summarize(name, v22, v115, taps):
    print(f"\n[{name}]")
    for lbl, v in (("22 kV (ผู้ใช้ไฟเห็น) ", v22), ("115 kV (สายซับทรานส์)", v115)):
        print(f"  {lbl}: ต่ำสุด {v.min():.4f} pu | "
              f"ตกครั้งเดียวมากสุด {np.abs(np.diff(v)).max() * 100:5.2f}% | "
              f"ต่ำกว่า {V_LIMIT} pu รวม {int((v < V_LIMIT).sum())} วินาที")
    print(f"  OLTC ขยับ {int((np.diff(taps) != 0).sum())} ครั้งใน "
          f"{len(taps) // 60} นาที (tap {taps.min()} ถึง {taps.max()})")


# ================================================== 5) วาดรูป
RED, BLUE, GREY, DARK = "#c1440e", "#1f6f8b", "#555555", "#8b1a1a"
n_bad = int((v115_s < V_LIMIT).sum())
n_bad_r = int((v115_r < V_LIMIT).sum())

fig = plt.figure(figsize=(10.5, 13.5))
gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 0.85, 1.25], hspace=0.30)

# --- แผงที่ 1: ผังระบบเส้นเดียว
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(0, 10)
ax0.set_ylim(-0.9, 4.0)
ax0.axis("off")
ax0.set_title("ระบบที่ใช้จำลอง (ระบบทดสอบสมมติ)", fontsize=14, pad=2)

xs = [0.8, 3.0, 6.0, 9.0]
names = ["ระบบส่ง 230 kV\n(slack)", "บัส 115 kV\nต้นทาง",
         "สถานีจำหน่าย\n115 kV", "Data Center\n115 kV"]
for x, nm, c in zip(xs, names, [GREY, GREY, BLUE, RED]):
    ax0.plot([x], [1.9], "o", ms=13, color=c, zorder=3)
    ax0.text(x, 2.12, nm, ha="center", va="bottom", fontsize=10.5,
             color=c, weight="bold")
ax0.plot([xs[0], xs[-1]], [1.9, 1.9], color="#333", lw=2.2, zorder=1)

ax0.text((xs[0] + xs[1]) / 2, 1.62, "หม้อแปลง 200 MVA\n230/115 kV · vk 12%",
         ha="center", va="top", fontsize=9.5, color=GREY)
ax0.text((xs[1] + xs[2]) / 2, 1.62,
         f"สาย 115 kV {L_SHARED} km (ใช้ร่วมกัน)\n"
         f"R = {LINE_R * L_SHARED:.1f} Ω · X = {LINE_X * L_SHARED:.1f} Ω",
         ha="center", va="top", fontsize=9.5, color=DARK)
ax0.text((xs[2] + xs[3]) / 2, 1.62, f"สาย 115 kV {L_SPUR} km (ช่วงแยกเข้า DC)",
         ha="center", va="top", fontsize=9.5, color="#333")

ax0.plot([xs[2], xs[2]], [0.30, 1.85], color=BLUE, lw=2)
ax0.plot([xs[2] - 0.9, xs[2] + 0.9], [0.30, 0.30], color=BLUE, lw=2)
ax0.plot([xs[2]], [0.30], "o", ms=11, color=BLUE)
ax0.text(xs[2] + 0.15, 0.78, "หม้อแปลง 50 MVA\n115/22 kV",
         ha="left", va="center", fontsize=9.5, color=BLUE)
ax0.text(xs[2], 0.12,
         f"บัส 22 kV — ชุมชน {NEIGHBOR_P:.0f} MW / {NEIGHBOR_Q:.0f} MVAr\n"
         f"มี OLTC {TAP_MIN} ถึง +{TAP_MAX} tap × {TAP_STEP}% · "
         f"หน่วง {REG_DELAY} วินาทีก่อนขยับแต่ละครั้ง",
         ha="center", va="top", fontsize=9.5, color=BLUE, weight="bold")
ax0.annotate("", xy=(xs[3], 1.55), xytext=(xs[3], 1.85),
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax0.text(xs[3], 1.45, f"{DC_IDLE:.0f} → {DC_FULL:.0f} MW",
         ha="center", va="top", fontsize=10, color=RED, weight="bold")
ax0.text(0.2, 3.95,
         "กระแสของ Data Center ไหลผ่านสาย 115 kV ช่วงที่สถานีจำหน่ายใช้ร่วมกัน",
         fontsize=10.5, color=DARK, va="top")

# --- แผงที่ 2: กำลังไฟฟ้าจริงของ Data Center
ax1 = fig.add_subplot(gs[1])
ax1.plot(t, p_s, color=RED, lw=2.2, label="โหลดจริงของ AI cluster")
ax1.plot(t, p_r, color=BLUE, lw=2.2, label=f"ถ้าบังคับ ramp {RAMP_LIMIT} MW/s")
ax1.set_ylabel("โหลด Data Center (MW)", fontsize=12)
ax1.set_ylim(0, 152)
ax1.grid(alpha=.3)
ax1.legend(fontsize=10.5, loc="lower right", framealpha=.95)
ax1.annotate(f"เริ่มงานเทรน  +{DC_FULL - DC_IDLE:.0f} MW ใน 1 วินาที",
             xy=(62, 114), xytext=(92, 138), fontsize=11, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
ax1.annotate("checkpoint — หยุดพร้อมกันทั้งคลัสเตอร์",
             xy=(315, 22), xytext=(340, 52), fontsize=11, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))

# --- แผงที่ 3: แรงดันที่บัส 115 kV
ax2 = fig.add_subplot(gs[2])
ax2.axhspan(0.938, V_LIMIT, color="#f3cdc4", alpha=.5)
ax2.fill_between(t, 0.938, V_LIMIT, where=v115_s < V_LIMIT, color=RED, alpha=.28)
ax2.plot(t, v115_s, lw=2.4, color=RED, label="โหลดจริงของ AI cluster")
ax2.plot(t, v115_r, lw=2.4, color=BLUE, label=f"ถ้าบังคับ ramp {RAMP_LIMIT} MW/s")
ax2.axhline(V_LIMIT, color=DARK, ls="--", lw=1.8)
ax2.text(6, V_LIMIT + 0.0015, f"เกณฑ์แรงดันต่ำสุด {V_LIMIT} pu",
         color=DARK, fontsize=11, weight="bold")
ax2.set_xlabel("เวลา (วินาที)", fontsize=12)
ax2.set_ylabel("แรงดันที่บัส 115 kV (pu)", fontsize=12)
ax2.set_ylim(0.938, 1.004)
ax2.grid(alpha=.3)
ax2.legend(fontsize=10.5, loc="lower right", framealpha=.95)
ax2.text(0.025, 0.95,
         f"แรงดันหลุดเกณฑ์ {n_bad} วินาที\n"
         f"ตกทันที {np.abs(np.diff(v115_s)).max() * 100:.2f}% ในหนึ่งวินาที\n"
         f"กรณีบังคับ ramp: หลุด {n_bad_r} วินาที",
         transform=ax2.transAxes, fontsize=12.5, ha="left", va="top",
         color=DARK, weight="bold",
         bbox=dict(boxstyle="round,pad=0.5", fc="#fdeeea", ec=DARK, lw=1.5))

fig.suptitle(f"โหลด {DC_FULL:.0f} MW เท่ากัน ต่างกันแค่ “มาถึงเร็วแค่ไหน”",
             fontsize=18, y=0.982, weight="bold")
fig.tight_layout(rect=[0, 0.008, 1, 0.955])
fig.savefig(OUT_PNG, dpi=140)


# ================================================== 6) สรุปผล
if __name__ == "__main__":
    print(f"Data Center {DC_IDLE:.0f} → {DC_FULL:.0f} MW ต่อที่ 115 kV | "
          f"สายช่วงใช้ร่วมกัน {L_SHARED} km | ชุมชน {NEIGHBOR_P:.0f} MW | "
          f"OLTC หน่วง {REG_DELAY} วินาที")
    summarize("โหลดจริงของ AI cluster (กระโดดทั้งก้อน)", v22_s, v115_s, tap_s)
    summarize(f"บังคับ ramp {RAMP_LIMIT} MW/s (พลังงานเท่ากัน)", v22_r, v115_r, tap_r)
    print(f"\nสรุป: บังคับ ramp rate แล้วเวลาที่บัส 115 kV หลุดเกณฑ์"
          f"ลดจาก {n_bad} เหลือ {n_bad_r} วินาที")
    print("      การต่อที่ 115 kV ช่วยผู้ใช้ไฟฝั่ง 22 kV ได้ (OLTC ดันกลับขึ้นมาทัน)")
    print("      แต่บัส 115 kV เองยังหลุดเกณฑ์ และ OLTC ต้องวิ่งทุกครั้งที่คลัสเตอร์")
    print("      เริ่มหรือหยุดงาน ซึ่งเป็นต้นทุนค่าสึกหรอที่ไม่ได้อยู่ในบิลค่าไฟใคร")
    print(f"\nรูปผลลัพธ์: {OUT_PNG}")
    print("\nลองแก้ต่อ: เปลี่ยน L_SHARED เพื่อหาความยาวสายสูงสุดที่ยังผ่านเกณฑ์ |"
          " เปลี่ยน DC_FULL เพื่อหาขนาดสูงสุดที่จุดนั้นรับได้ |"
          " เปลี่ยน REG_DELAY เพื่อดูว่าตั้งเร็วขึ้นแล้ว tap ต้องวิ่งเพิ่มเท่าไร")
    print("\nระบบทดสอบสมมติทั้งหมด — ตัวอย่างเพื่อการเรียนรู้")
    print("งานจริงต้องตรวจสอบโดยวิศวกรผู้มีใบอนุญาต")
    if IN_NOTEBOOK:
        plt.show()
