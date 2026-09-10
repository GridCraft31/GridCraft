#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ระบบแรงต่ำ: โซลาร์เที่ยงดันแรงดันเกิน · EV หัวค่ำดึงแรงดันตก · แบตช่วยยังไง
================================================================================

โจทย์
    หม้อแปลงจำหน่าย 22/0.4 kV 250 kVA จ่ายสาย LV สองเส้น
      * สาย A ยาว 400 m 10 บัส (เสาทุก 40 m) · สาย B ยาว 240 m 6 บัส
      * บ้าน 4 หลังต่อบัส รวม 64 หลัง · ครึ่งหนึ่งติดโซลาร์หลังคา 5 kWp
      * EV 5 คัน ชาร์จ 7.4 kW เวลาเสียบสุ่ม (seed คงที่ รันซ้ำได้ผลเดิม)
    จำลอง 1 วัน ราย 5 นาที (288 จุด) · แดดแบบฟ้าใส (clear sky)

    คำถาม: เที่ยงแรงดันเกิน หัวค่ำแรงดันตก บนสายเส้นเดียวกันในวันเดียวกัน
            ปรับ tap หม้อแปลงแก้ได้ไหม และถ้าใช้แบต ต้องใหญ่แค่ไหน

วิธีทำ
    quasi-static power flow ราย 5 นาที (pandapower)

วิธีรัน
    pip install pandapower matplotlib
    python post02_share.py

    บน Google Colab: !pip install -q pandapower แล้วก๊อปไฟล์นี้ทั้งไฟล์วางในเซลล์เดียว
    ใช้เวลาราว 3–4 นาที (รันทั้งหมด 7 วัน-จำลอง)

ที่มาของตัวเลข
    [1] ANSI C84.1 Range A — กรอบแรงดันบริการ ±5% ที่ IEEE Std 1547-2018 อ้างถึง
        NREL, "An Overview of Issues Related to IEEE Std 1547-2018"
        https://docs.nrel.gov/docs/fy21osti/77156.pdf
    [2] IEC 60076-5 Table 1 — หม้อแปลง 25–630 kVA ค่า short-circuit impedance
        ขั้นต่ำที่ยอมรับทั่วไป 4.0%
    [3] สาย ABC 4 แกน อะลูมิเนียม 95 mm² 0.6/1 kV (datasheet APEC, AS/NZS 3560.1)
        R ac 0.398 Ω/km · X 0.0868 Ω/km · พิกัดกระแส 216 A ที่ลม 1 m/s
        https://media.mmem.com.au/Datasheets/APEC/APEC_Alum_Aerial_Bundled_4Core.pdf
    [4] แบบจำลองแดดฟ้าใส Haurwitz (1945) ตามที่ pvlib ใช้
        GHI = 1098 · cos(z) · exp(−0.059 / cos(z))  W/m²
        https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.clearsky.haurwitz.html
    [5] ที่ชาร์จ EV ในบ้านแบบเฟสเดียว 32 A × 230 V ≈ 7.4 kW

ค่าที่เป็นสมมติฐาน (ผลไวต่อค่าเหล่านี้)
    - รูปโหลดบ้านรายวันเป็นรูปสังเคราะห์ ไม่ใช่ข้อมูลวัดจริง
    - PV คิดจาก GHI แนวราบ × performance ratio 0.8 (ไม่คิดมุมเอียงแผง)
    - เวลาเสียบ EV สุ่มรอบ 18:30 น. · พลังงานที่ต้องเติมสุ่ม 10–30 kWh
    - โมเดลเป็น 3 เฟสสมดุล — ของจริงบ้านเป็นเฟสเดียว เฟสที่หนักที่สุดจะแย่กว่านี้

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

# ---- ฟอนต์ไทย -------------------------------------------------------------
# Colab ไม่มีฟอนต์ไทยติดมา ถ้าไม่มีในเครื่องจะโหลด Noto Sans Thai มาเก็บข้างไฟล์นี้
# ต้องตั้ง font.family เป็น list เพื่อ fallback ไปหา DejaVu สำหรับ Ω และ →
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
DT_MIN = 5
STEPS = 24 * 60 // DT_MIN            # 288 จุด = 1 วัน
DT_H = DT_MIN / 60

# ---- หม้อแปลง 22/0.4 kV
TR_KVA = 250
TR_VK, TR_VKR = 4.0, 1.3             # %  vk ตาม IEC 60076-5 [2] · vkr สมมติ
TAP_STEP = 2.5                       # %  off-load tap ขั้นละ 2.5%
TAP_POS = -1                         # −1 = ยกฝั่ง LV ขึ้น 2.5% (ลด tap ฝั่งสูง = LV สูงขึ้น)

# ---- สาย LV
LINE_R, LINE_X = 0.398, 0.0868       # Ω/km  ABC 4×95 mm² [3]
LINE_IMAX = 0.216                    # kA
SPAN_M = 40                          # ระยะเสา
N_BUS_A, N_BUS_B = 10, 6
HOUSES_PER_BUS = 4

# ---- โหลดบ้าน
HOUSE_PEAK_KW = 2.0                  # พีคหัวค่ำต่อหลัง (หลังกระจายตัวแล้ว)
PF = 0.95

# ---- โซลาร์หลังคา
LAT, LON, TZ_H = 13.75, 100.5, 7     # ภาคกลางของไทย
DAY_OF_YEAR = 105                    # 15 เม.ย. — แดดเกือบตั้งฉากตอนเที่ยง
PV_KWP = 5.0                         # ต่อหลัง
PV_SHARE = 0.5                       # สัดส่วนบ้านที่ติดโซลาร์
PR = 0.8

# ---- EV
N_EV = 5
EV_KW = 7.4                          # [5]
EV_SEED = 7                          # เปลี่ยน seed = สุ่มชุดใหม่
EV_ARRIVE_MEAN, EV_ARRIVE_SD = 18.5, 1.0
EV_KWH_RANGE = (10, 30)

# ---- เกณฑ์และแบต
V_LOW, V_HIGH = 0.95, 1.05           # [1]
BESS_KW = 30.0
BESS_KWH = 120.0                     # ขนาดเล็กที่สุดที่ผ่านในตารางที่ลองไล่ดู
V_DB_HI, V_DB_LO = 1.03, 0.97        # เริ่ม droop ตั้งแต่ออกนอกช่วงนี้


# ================================================== 1) รูปรายวัน
def hours():
    return np.arange(STEPS) * DT_H


def clear_sky_pu():
    """กำลัง PV ต่อ kWp จากแดดฟ้าใส Haurwitz [4] + ตำแหน่งดวงอาทิตย์สูตรมาตรฐาน"""
    h = hours() + DT_H / 2
    g = 2 * np.pi / 365 * (DAY_OF_YEAR - 1 + (h - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    solar_min = h * 60 + eot + 4 * LON - 60 * TZ_H
    ha = np.radians(solar_min / 4 - 180)
    lat = np.radians(LAT)
    cz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    ghi = np.where(cz > 0, 1098.0 * cz * np.exp(-0.059 / np.maximum(cz, 1e-6)), 0.0)
    return ghi / 1000.0 * PR


def house_shape():
    """โหลดบ้าน (pu ของพีค): ดึกต่ำ · เช้าขึ้นนิด · กลางวันเงียบ · หัวค่ำพีค ~20:00"""
    h = hours()
    night = 0.30
    morning = 0.25 * np.exp(-((h - 7.0) ** 2) / (2 * 1.0 ** 2))
    day = 0.10 * np.exp(-((h - 13.0) ** 2) / (2 * 3.0 ** 2))
    evening = 0.70 * np.exp(-((h - 20.0) ** 2) / (2 * 1.6 ** 2))
    return np.clip(night + morning + day + evening, 0, 1.0)


def ev_sessions(buses):
    """สุ่ม EV: บัสที่จอด · เวลาเสียบ · ต้องชาร์จนานเท่าไร (seed คงที่)"""
    rng = np.random.default_rng(EV_SEED)
    at = rng.choice(len(buses), size=N_EV, replace=False)
    arrive = np.clip(rng.normal(EV_ARRIVE_MEAN, EV_ARRIVE_SD, N_EV), 16.5, 22.5)
    need = rng.uniform(*EV_KWH_RANGE, N_EV)
    return [dict(bus=buses[i], arrive=a, hours=e / EV_KW, kwh=e)
            for i, a, e in zip(at, arrive, need)]


def ev_profile(s):
    h = hours()
    return np.where((h >= s["arrive"]) & (h < s["arrive"] + s["hours"]), EV_KW, 0.0)


# ================================================== 2) สร้างระบบ
def build(tap_pos=TAP_POS):
    net = pp.create_empty_network()
    bmv = pp.create_bus(net, vn_kv=22.0, name="22 kV")
    pp.create_ext_grid(net, bmv, vm_pu=1.0)
    blv = pp.create_bus(net, vn_kv=0.4, name="บัส LV หม้อแปลง")

    # หมายเหตุ pandapower 3.x: ถ้าไม่ใส่ tap_changer_type="Ratio"
    # ค่า tap_pos จะถูกเมินเงียบ ๆ power flow รันผ่านแต่แรงดันไม่ขยับเลย
    pp.create_transformer_from_parameters(
        net, bmv, blv, sn_mva=TR_KVA / 1000, vn_hv_kv=22.0, vn_lv_kv=0.4,
        vk_percent=TR_VK, vkr_percent=TR_VKR, pfe_kw=0.6, i0_percent=0.3,
        tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2,
        tap_step_percent=TAP_STEP, tap_pos=tap_pos, tap_changer_type="Ratio")

    feeders = {}
    for name, n in (("A", N_BUS_A), ("B", N_BUS_B)):
        prev, buses = blv, []
        for k in range(1, n + 1):
            b = pp.create_bus(net, vn_kv=0.4, name=f"{name}{k} ({k * SPAN_M} m)")
            pp.create_line_from_parameters(
                net, prev, b, length_km=SPAN_M / 1000, r_ohm_per_km=LINE_R,
                x_ohm_per_km=LINE_X, c_nf_per_km=0, max_i_ka=LINE_IMAX)
            buses.append(b)
            prev = b
        feeders[name] = buses
    return net, blv, feeders


# ================================================== 3) รันหนึ่งวัน
def simulate(with_pv=True, with_ev=True, with_bess=False, tap_pos=TAP_POS):
    net, blv, feeders = build(tap_pos)
    all_buses = feeders["A"] + feeders["B"]

    pv_per_bus = HOUSES_PER_BUS * PV_SHARE * PV_KWP
    loads = {b: pp.create_load(net, b, p_mw=0, q_mvar=0) for b in all_buses}
    pvs = {b: pp.create_sgen(net, b, p_mw=0, q_mvar=0) for b in all_buses}

    # EV สุ่มเฉพาะบ้านบนสาย A (สายยาวที่ปัญหาจะโผล่จริง)
    sessions = ev_sessions(feeders["A"]) if with_ev else []
    for s in sessions:
        s["name"] = net.bus.name.at[s["bus"]]
    ev_kw = {b: np.zeros(STEPS) for b in all_buses}
    for s in sessions:
        ev_kw[s["bus"]] += ev_profile(s)

    end = feeders["A"][-1]
    bess = pp.create_sgen(net, end, p_mw=0, q_mvar=0, name="BESS")

    hs, pv = house_shape(), clear_sky_pu()
    tanphi = np.tan(np.arccos(PF))
    V = np.zeros((STEPS, len(all_buses)))
    v_lv = np.zeros(STEPS)
    p_b, soc = np.zeros(STEPS), np.zeros(STEPS)
    e = 0.0

    for t in range(STEPS):
        for b in all_buses:
            house = HOUSES_PER_BUS * HOUSE_PEAK_KW * hs[t]
            net.load.at[loads[b], "p_mw"] = (house + ev_kw[b][t]) / 1000
            net.load.at[loads[b], "q_mvar"] = house * tanphi / 1000
            net.sgen.at[pvs[b], "p_mw"] = pv_per_bus * pv[t] / 1000 if with_pv else 0.0
        net.sgen.at[bess, "p_mw"] = 0.0
        pp.runpp(net)

        if with_bess:
            v = net.res_bus.vm_pu.at[end]
            pk = 0.0
            # droop แบบเส้นตรง (รูปเดียวกับ Volt-Watt ใน IEEE 1547) ไม่ใช่เปิด/ปิด
            # ถ้าเปิด/ปิด แบตจะชาร์จเต็มกำลังตั้งแต่สาย ๆ แล้วเต็มก่อนเที่ยง
            if v > V_DB_HI and e < BESS_KWH:
                k = min((v - V_DB_HI) / (V_HIGH - V_DB_HI), 1.0)
                pk = -min(BESS_KW * k, (BESS_KWH - e) / DT_H)
            elif v < V_DB_LO and e > 0:
                k = min((V_DB_LO - v) / (V_DB_LO - V_LOW), 1.0)
                pk = min(BESS_KW * k, e / DT_H)
            if pk:
                net.sgen.at[bess, "p_mw"] = pk / 1000
                pp.runpp(net)
            p_b[t] = pk
            e = float(np.clip(e - pk * DT_H, 0, BESS_KWH))
        soc[t] = e

        V[t] = net.res_bus.vm_pu.loc[all_buses].values
        v_lv[t] = net.res_bus.vm_pu.at[blv]

    return dict(V=V, v_lv=v_lv, sessions=sessions, p_bess=p_b, soc=soc,
                pv_kw=pv_per_bus * len(all_buses) * pv,
                names=[net.bus.name.at[b] for b in all_buses])


def minutes(v, cond):
    return int(cond(v).sum()) * DT_MIN


# ================================================== 4) รันทุกกรณี
print("กำลังจำลอง 1 วัน ราย 5 นาที …")
h = hours()
r_prob = simulate(True, True)                          # โซลาร์ + EV ไม่มีแบต
r_fix = simulate(True, True, with_bess=True)           # + แบต
taps = (-2, -1, 0, 1, 2)
r_taps = {tp: simulate(True, True, tap_pos=tp) for tp in taps}

end = N_BUS_A - 1                                      # บ้านหลังสุดท้ายของสาย A
n_house = (N_BUS_A + N_BUS_B) * HOUSES_PER_BUS
house_kw = n_house * HOUSE_PEAK_KW * house_shape()
ev_kw = np.zeros_like(h)
for s in r_prob["sessions"]:
    ev_kw += ev_profile(s)
pv_kw = r_prob["pv_kw"]
v_p, v_f = r_prob["V"][:, end], r_fix["V"][:, end]
hi_p, lo_p = minutes(v_p, lambda v: v > V_HIGH), minutes(v_p, lambda v: v < V_LOW)
hi_f, lo_f = minutes(v_f, lambda v: v > V_HIGH), minutes(v_f, lambda v: v < V_LOW)

# ================================================== 5) รูปผลลัพธ์
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK2,
})


def band(ax):
    for y in (V_LOW, V_HIGH):
        ax.axhline(y, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)


# ---- รูปที่ 1: ภาพรวมทั้งวัน
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True,
                                 gridspec_kw={"height_ratios": [1.0, 1.45, 0.75]})
a1.plot(h, house_kw, color=INK2, label=f"โหลดบ้าน {n_house} หลัง")
a1.fill_between(h, 0, pv_kw, color=YELLOW, alpha=0.10, lw=0)
a1.plot(h, pv_kw, color=YELLOW, label=f"โซลาร์หลังคา {n_house * PV_SHARE * PV_KWP:.0f} kWp (ฟ้าใส)")
a1.plot(h, ev_kw, color=AQUA, label=f"EV {N_EV} คัน × {EV_KW} kW")
a1.set_ylabel("กำลังไฟ (kW)")
a1.legend(loc="upper left", fontsize=9)
a1.set_ylim(0, max(pv_kw.max(), house_kw.max() + ev_kw.max()) * 1.25)

band(a2)
a2.fill_between(h, v_p, V_HIGH, where=v_p > V_HIGH, color=ORANGE, alpha=0.12, lw=0)
a2.fill_between(h, v_p, V_LOW, where=v_p < V_LOW, color=ORANGE, alpha=0.12, lw=0)
a2.plot(h, v_p, color=ORANGE, label="ไม่มีแบต")
a2.plot(h, v_f, color=BLUE, label="มีแบตที่ปลายสาย")
a2.set_ylabel("แรงดันบ้านหลังสุดท้าย (pu)")
a2.legend(loc="upper right", fontsize=9)
i_hi, i_lo = int(np.argmax(v_p)), int(np.argmin(v_p))
a2.annotate(f"เกิน {V_HIGH} รวม {hi_p} นาที", xy=(h[i_hi], v_p[i_hi]),
            xytext=(h[i_hi] - 5.5, v_p[i_hi] + 0.012), fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
a2.annotate(f"ต่ำกว่า {V_LOW} รวม {lo_p} นาที", xy=(h[i_lo], v_p[i_lo]),
            xytext=(h[i_lo] - 7.5, v_p[i_lo] - 0.012), fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
# เส้นน้ำเงินกระโดดตอนบ่าย = แบตเต็ม ดูดต่อไม่ได้ ไม่ใช่บั๊ก
full = np.flatnonzero(r_fix["soc"] >= BESS_KWH - 1e-6)
if full.size:
    tf = int(full[0])
    hh, mm = divmod(int(round(h[tf] * 60)), 60)
    a2.annotate(f"แบตเต็ม {hh:02d}:{mm:02d}", xy=(h[tf], v_f[min(tf + 1, STEPS - 1)]),
                xytext=(h[tf] + 1.2, v_f[tf] - 0.012), fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
a2.set_ylim(min(v_p.min(), v_f.min()) - 0.025, max(v_p.max(), v_f.max()) + 0.025)

pb = r_fix["p_bess"]
a3.axhline(0, color=AXIS, lw=1.0)
a3.fill_between(h, 0, pb, color=BLUE, alpha=0.10, lw=0)
a3.plot(h, pb, color=BLUE)
a3.set_ylabel("แบต (kW)")
lim = max(abs(pb).max(), 1) * 1.45
a3.set_ylim(-lim, lim)
a3.text(13.9, -lim * 0.45, "← ดูดไฟเก็บตอนแดดแรง", ha="left", va="center", fontsize=9.5, color=INK2)
a3.text(17.7, lim * 0.45, "คายคืนตอน EV เสียบ →", ha="right", va="center", fontsize=9.5, color=INK2)
a3.set_xlim(0, 24)
a3.set_xticks(range(0, 25, 3))
a3.set_xticklabels([f"{x:02d}:00" for x in range(0, 25, 3)])
a3.set_xlabel("เวลา")
fig.suptitle("วันเดียว สายเส้นเดียว เจอทั้งแรงดันเกินและแรงดันตก",
             fontsize=15, weight="bold", y=0.99)
fig.text(0.5, 0.932, f"หม้อแปลง {TR_KVA} kVA · สาย LV {N_BUS_A * SPAN_M} m · แบต {BESS_KW:.0f} kW / "
         f"{BESS_KWH:.0f} kWh → หลุดเกณฑ์เหลือ {hi_f + lo_f} นาที", ha="center", fontsize=10, color=INK2)
fig.tight_layout(rect=[0, 0, 1, 0.915])
fig.savefig("lv_solar_ev_bess.png", dpi=150)

# ---- รูปที่ 2: แรงดันไล่ตามระยะสาย A
x_m = np.arange(0, N_BUS_A + 1) * SPAN_M
fig2, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, t, title in ((axes[0], i_hi, "ช่วงแดดแรง"), (axes[1], i_lo, "ช่วง EV เสียบ")):
    band(ax)
    for r, col, lab in ((r_prob, ORANGE, "ไม่มีแบต"), (r_fix, BLUE, "มีแบตที่ปลายสาย")):
        v = np.r_[r["v_lv"][t], r["V"][t, :N_BUS_A]]
        ax.plot(x_m, v, color=col, label=lab, marker="o", ms=4.5,
                markeredgecolor=SURF, markeredgewidth=1.5)
    hh, mm = divmod(int(round(h[t] * 60)), 60)
    ax.set_title(f"{title} {hh:02d}:{mm:02d} น.", fontsize=11.5, loc="left")
    ax.set_xlabel("ระยะจากหม้อแปลงตามสาย A (m)")
    ax.set_xticks(x_m[::2])
axes[0].set_ylabel("แรงดัน (pu)")
axes[1].legend(loc="lower left", fontsize=9)
fig2.suptitle("ยิ่งไกลหม้อแปลง แรงดันยิ่งแกว่ง — ขึ้นตอนเที่ยง ลงตอนหัวค่ำ",
              fontsize=13, weight="bold", x=0.02, ha="left")
fig2.tight_layout()
fig2.savefig("lv_profile.png", dpi=150)

# ---- รูปที่ 3: ปรับ tap อย่างเดียว vs มีแบต
rows = []
for tp in taps:
    Vt = r_taps[tp]["V"]
    rows.append((f"tap {tp * -TAP_STEP:+.1f}%",
                 minutes(Vt.min(axis=1), lambda v: v < V_LOW),
                 minutes(Vt.max(axis=1), lambda v: v > V_HIGH)))
Vf = r_fix["V"]
rows.append((f"tap {TAP_POS * -TAP_STEP:+.1f}% + แบต",
             minutes(Vf.min(axis=1), lambda v: v < V_LOW),
             minutes(Vf.max(axis=1), lambda v: v > V_HIGH)))
fig3, ax = plt.subplots(figsize=(8.4, 4.6))
y = np.arange(len(rows))[::-1]
for yi, (lab, under, over) in zip(y, rows):
    ax.barh(yi, -under, height=0.5, color=BLUE)
    ax.barh(yi, over, height=0.5, color=RED)
    if under:
        ax.text(-under - 8, yi, f"{under}", va="center", ha="right", fontsize=9.5, color=INK2)
    if over:
        ax.text(over + 8, yi, f"{over}", va="center", ha="left", fontsize=9.5, color=INK2)
    if not under and not over:
        ax.text(8, yi, "0 นาที", va="center", ha="left", fontsize=9.5, weight="bold")
ax.axvline(0, color=AXIS, lw=1)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
m = max(max(r[1], r[2]) for r in rows) * 1.25
ax.set_xlim(-m, m)
ticks = ax.get_xticks()
ax.set_xticks(ticks)
ax.set_xticklabels([f"{abs(int(t))}" for t in ticks])
ax.set_xlabel("← นาทีที่แรงดันต่ำกว่า 0.95        นาทีที่แรงดันเกิน 1.05 →")
ax.grid(axis="y", visible=False)
ax.set_title("ปรับ tap หม้อแปลงอย่างเดียว = ย้ายปัญหา ไม่ใช่แก้ปัญหา",
             fontsize=13, weight="bold", loc="left")
fig3.tight_layout()
fig3.savefig("lv_tap.png", dpi=150)

# ================================================== 6) สรุปผล
print(f"\nหม้อแปลง {TR_KVA} kVA · บ้าน {n_house} หลัง · PV {n_house * PV_SHARE * PV_KWP:.0f} kWp "
      f"· EV {N_EV} คัน × {EV_KW} kW")
print("EV ที่สุ่มได้:")
for s in sorted(r_prob["sessions"], key=lambda s: s["arrive"]):
    hh, mm = int(s["arrive"]), int(round((s["arrive"] % 1) * 60))
    print(f"   เสียบ {hh:02d}:{mm:02d} น. · บ้านบัส {s['name']} · เติม {s['kwh']:.1f} kWh")
print(f"\nแรงดันบ้านหลังสุดท้าย (A10) ไม่มีแบต: {v_p.min():.4f}–{v_p.max():.4f} pu "
      f"· เกิน {hi_p} นาที · ตก {lo_p} นาที")
print(f"มีแบต {BESS_KW:.0f} kW / {BESS_KWH:.0f} kWh: {v_f.min():.4f}–{v_f.max():.4f} pu "
      f"· เกิน {hi_f} นาที · ตก {lo_f} นาที")
print("\nปรับ tap อย่างเดียว (ทุกบัส):")
for lab, under, over in rows:
    print(f"   {lab:<18} ต่ำ {under:>4d} นาที · เกิน {over:>4d} นาที")

if IN_NOTEBOOK:
    plt.show()
