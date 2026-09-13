#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ระบบแรงต่ำที่มีโซลาร์หลังคาและการอัดประจุ EV: Monte Carlo แบบ 3 เฟส 4 สาย
================================================================================

กรณีศึกษา (ระบบทดสอบสมมติ)
    หม้อแปลงจำหน่าย 250 kVA 22/0.4 kV Dyn11 จ่ายวงจรแรงต่ำสองวงจร
      วงจร 1 ยาว 300 m 10 เสา, วงจร 2 ยาว 180 m 6 เสา
      ผู้ใช้ไฟ 48 ราย เสาละ 3 ราย (เฉลี่ยเฟสละ 1 ราย) ต่อวนเฟส A B C
      โหลดพีค 3 kW ต่อราย สุ่มขนาดและเวลาที่เกิดพีครายราย ได้ราว 16 kWh ต่อวัน
      โซลาร์หลังคา 24 ราย รายละ 5 kWp, เครื่องอัดประจุ EV 14 จุด จุดละ 7.4 kW
      ท้องฟ้าแจ่มใส, ความละเอียดข้อมูล 5 นาที, 1 วัน
    ผู้ใช้ไฟทุกรายต่อแบบเฟสเดียว โซลาร์และเครื่องอัดประจุของรายนั้นจึงอยู่บนเฟสเดียวกัน
    สุ่มตำแหน่งผู้ติดตั้งโซลาร์ ตำแหน่งเครื่องอัดประจุ และเวลาเริ่มอัดประจุ ทำซ้ำ N_DRAW กรณี
    แล้วเปรียบเทียบกับแบบจำลอง 3 เฟสสมดุล (เกลี่ยโหลดของแต่ละเสาเท่ากันทุกเฟส) บนชุดตัวอย่างเดียวกัน

วิธีคำนวณ
    power flow ไม่สมดุลด้วย pandapower (runpp_3ph) โหลดต่อเฟส-นิวทรัลรายเสา
    สายคำนวณจากเมทริกซ์อิมพีแดนซ์ 4x4 แบบ Carson แล้วยุบตัวนำนิวทรัล ได้ Z1 ตรงกับ datasheet และ Z0 = 4 x Z1
    (ค่า Z0 นี้ใช้ได้เฉพาะกรณีนิวทรัลต่อลงดินที่หม้อแปลงจุดเดียว กระแสจึงไหลกลับทางนิวทรัลทั้งหมด)
    ในไฟล์มี solver แบบ backward forward sweep 4 ตัวนำด้วย (ENGINE = "fast")
    ทั้งสองวิธีให้ผลตรงกันในระดับ 0.002 pu แบบ fast เร็วกว่ามาก เหมาะกับการสุ่มระดับพันกรณี
    เกณฑ์ตัดสินใช้แรงดันเฟสเทียบนิวทรัลที่ผู้ใช้ไฟได้รับ

วิธีรัน
    pip install pandapower numpy matplotlib
    python post02_montecarlo.py

    ค่าเริ่มต้น ENGINE = "pandapower" สุ่ม 300 กรณี คำนวณทุก 10 นาที ใช้เวลาราว 75 นาที
    ตั้ง ENGINE = "fast" จะสุ่ม 2,000 กรณี คำนวณทุก 5 นาที ใช้เวลาไม่กี่นาที

ที่มาของตัวเลข
    [1] IEC 60038 และ EN 50160 แรงดันระบบแรงต่ำ บวกลบ 10% ของแรงดันระบุ
        (EN 50160 ใช้กับค่าเฉลี่ย 10 นาที)
    [2] IEC 60076-5 หม้อแปลง 25-630 kVA ค่า short-circuit impedance ขั้นต่ำ 4.0%
    [3] สาย ABC 4x95 mm2 อะลูมิเนียม R 0.398, X 0.0868 ohm/km (datasheet APEC)
        https://media.mmem.com.au/Datasheets/APEC/APEC_Alum_Aerial_Bundled_4Core.pdf
    [4] แบบจำลองท้องฟ้าแจ่มใส Haurwitz ตามที่ pvlib ใช้
        https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.clearsky.haurwitz.html
    [5] เครื่องอัดประจุ EV เฟสเดียว 32 A x 230 V ประมาณ 7.4 kW

สมมติฐาน (ผลไวต่อข้อเหล่านี้)
    - นิวทรัลต่อลงดินเฉพาะที่หม้อแปลง ไม่มีการต่อลงดินซ้ำตามเสา
    - ไม่คิดสายเข้าบ้าน, รูปโหลดรายวันเป็นแบบสังเคราะห์, ท้องฟ้าแจ่มใสทั้งวัน
    - ทุกเสามีตัวนำ A B C N ผู้ใช้ไฟต่อระหว่างเฟสหนึ่งกับ N วนเฟส A B C (PHASE_MODE = "random" = สุ่มเฟส)
    - ไม่รวมระบบกักเก็บพลังงาน

ระบบทั้งหมดเป็นระบบทดสอบสมมติ ไม่ใช่ข้อมูลของหน่วยงานใด
จัดทำเพื่อการเรียนรู้ การนำไปใช้งานจริงต้องตรวจสอบโดยวิศวกรผู้มีใบอนุญาต
================================================================================
"""
import os
import time
import matplotlib

try:
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None
except Exception:
    IN_NOTEBOOK = False
if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Circle
import numpy as np

# ---- ฟอนต์ไทย (Colab ไม่มีติดมา จะโหลด Noto Sans Thai ให้)
LOCAL_FONTS = ["/usr/share/fonts/truetype/tlwg/Garuda.ttf", "/usr/share/fonts/truetype/tlwg/Loma.ttf",
               "/usr/share/fonts/truetype/thai/Garuda.ttf", "NotoSansThai-Regular.ttf"]
NOTO_URL = ("https://github.com/googlefonts/noto-fonts/raw/main/"
            "hinted/ttf/NotoSansThai/NotoSansThai-Regular.ttf")


def _use_thai_font():
    path = next((f for f in LOCAL_FONTS if os.path.exists(f)), None)
    if path is None:
        try:
            import urllib.request
            path = "NotoSansThai-Regular.ttf"
            urllib.request.urlretrieve(NOTO_URL, path)
        except Exception as e:
            print(f"เตือน: หาฟอนต์ไทยไม่ได้ ({e}) ตัวหนังสือไทยจะเป็นสี่เหลี่ยม")
            return
    font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=path).get_name(), "DejaVu Sans"]


_use_thai_font()

# =============================================================== ตัวแปรที่ลองแก้ได้
ENGINE = "pandapower"          # "pandapower" = ใช้ runpp_3ph, "fast" = solver ในไฟล์นี้ (เร็วกว่ามาก)
N_DRAW = 300 if ENGINE == "pandapower" else 2000     # จำนวนกรณีที่สุ่ม
STEP_EVERY = 2 if ENGINE == "pandapower" else 1      # ใช้ทุกกี่จุดเวลา (2 = ราย 10 นาที)
SEED = 2026                    # เปลี่ยน seed = สุ่มชุดใหม่
N_PV = 24                      # จำนวนผู้ใช้ไฟที่ติดโซลาร์ (จาก 48 ราย)
N_EV = 14                      # จำนวนผู้ใช้ไฟที่มีเครื่องอัดประจุ EV
PHASE_MODE = "rotate"          # "rotate" = วนเฟส A B C ตามลำดับบ้าน, "random" = สุ่มเฟส
RUN_HOSTING = ENGINE == "fast"  # ตาราง hosting capacity (ช้าเกินถ้าใช้ pandapower)
N_HOST = 200                   # จำนวนกรณีต่อช่องในตาราง hosting

# =============================================================== ค่าระบบ
DT_MIN = 5
STEPS = 24 * 60 // DT_MIN
DT_H = DT_MIN / 60
TR_KVA, TR_VK, TR_VKR = 250, 4.0, 1.3        # [2]
TAP_STEP, TAP_POS = 2.5, -1                  # tap -1 = ยกฝั่ง LV ขึ้น 2.5%
PFE_KW, I0_PCT = 0.6, 0.3
LINE_R, LINE_X, SPAN_M = 0.398, 0.0868, 30   # [3]
N_A, N_B, HPB = 10, 6, 3                     # เสาละ 3 หลัง = เฟสละ 1 หลังต่อเสา
N_BUS, N_HOUSE = N_A + N_B, (N_A + N_B) * HPB
HOUSE_PEAK_KW, PF = 3.0, 0.95                # พีคของบ้านแต่ละหลัง
HOUSE_SCALE = (0.45, 1.0)                    # บ้านไม่เท่ากัน: ช่วงขนาดพีค (คูณกับ HOUSE_PEAK_KW)
HOUSE_SHIFT_SD = 0.75                        # ชั่วโมง: บ้านไม่ได้พีคพร้อมกัน
LAT, LON, TZ_H, DAY_OF_YEAR = 13.75, 100.5, 7, 105
PV_KWP, PR = 5.0, 0.8
EV_KW = 7.4                                  # [5]
EV_ARRIVE_MEAN, EV_ARRIVE_SD = 18.5, 1.0
EV_KWH_RANGE = (10, 30)
V_LOW, V_HIGH = 0.90, 1.10                   # [1]

# ---- หม้อแปลง Dyn11: ลำดับศูนย์ = leakage impedance จึงใส่ z ต่อเฟสได้ตรง ๆ
VPH = 400 / np.sqrt(3)
A = np.exp(2j * np.pi / 3)
E = VPH / (1 + TAP_POS * TAP_STEP / 100) * np.array([1, A ** 2, A])
_zb = 0.4 ** 2 / (TR_KVA / 1000)
Z_TR = complex(TR_VKR / 100 * _zb, np.sqrt((TR_VK / 100) ** 2 - (TR_VKR / 100) ** 2) * _zb)
S_MAG_PH = complex(PFE_KW, np.sqrt((I0_PCT / 100 * TR_KVA) ** 2 - PFE_KW ** 2)) * 1000 / 3

# ---- สาย ABC 3 เฟส + นิวทรัลขนาดเท่ากัน: เมทริกซ์ 4x4 แบบ Carson
# กระแสกลับทางนิวทรัลทั้งหมด (ต่อดินที่หม้อแปลงจุดเดียว) เทอม earth return จึงหักล้างกันหมด
# ระยะแกนถึงแกน 15 mm ทุกคู่ แล้วเลือก GMR ให้ Z1 = R + jX ตาม datasheet
W_LN = 2 * np.pi * 50 * 2e-7 * 1000          # ohm/km ต่อหน่วย ln
D_CORE = 0.015
GMR = D_CORE * np.exp(-LINE_X / W_LN)
Z_KM = np.full((4, 4), 1j * W_LN * np.log(1 / D_CORE))
np.fill_diagonal(Z_KM, LINE_R + 1j * W_LN * np.log(1 / GMR))
Z_SPAN = Z_KM * SPAN_M / 1000
# วงจรเฟสเดียว (ไปทางเฟส กลับทางนิวทรัล) = Zaa + Znn - 2Zan = 2 x Z1


# =============================================================== 1) รูปรายวัน
def hours():
    return np.arange(STEPS) * DT_H


def clear_sky_pu():
    """กำลัง PV ต่อ kWp จากแดดฟ้าใส Haurwitz [4]"""
    h = hours() + DT_H / 2
    g = 2 * np.pi / 365 * (DAY_OF_YEAR - 1 + (h - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.radians((h * 60 + eot + 4 * LON - 60 * TZ_H) / 4 - 180)
    lat = np.radians(LAT)
    cz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    ghi = np.where(cz > 0, 1098.0 * cz * np.exp(-0.059 / np.maximum(cz, 1e-6)), 0.0)
    return ghi / 1000.0 * PR


def house_profiles(scale, shift):
    """โหลดรายหลัง (m, T, จำนวนบ้าน) kW
    รูปวัน: ดึกต่ำ เช้าขึ้นนิด กลางวันเงียบ หัวค่ำพีคราว 20:00
    แต่ละหลังขนาดพีคไม่เท่ากันและพีคไม่พร้อมกัน (ของจริงบ้านไม่ได้ใช้ไฟเหมือนกันเป๊ะ)"""
    h = hours()[None, :, None]
    sh = shift[:, None, :]
    prof = np.clip(0.12 + 0.18 * np.exp(-((h - 7.0 - sh) ** 2) / 2)
                   + 0.07 * np.exp(-((h - 13.0 - sh) ** 2) / 18)
                   + 0.88 * np.exp(-((h - 20.0 - sh) ** 2) / (2 * 1.6 ** 2)), 0, 1.0)
    return prof * (scale[:, None, :] * HOUSE_PEAK_KW)


# =============================================================== 2) power flow 3 เฟส 4 สาย
def solve(S, tol=1e-6, it_max=80):
    """S (..., 16, 3) VA ต่อบัสต่อเฟส (บวก = ดึงไฟ)
    คืน V (..., 17, 4) โวลต์, โหนด 0 = บัส LV หม้อแปลง (นิวทรัล = 0 เพราะต่อดิน), คอลัมน์ 3 = นิวทรัล"""
    V = np.zeros(S.shape[:-2] + (N_BUS + 1, 4), complex)
    V[..., :3] = E
    for _ in range(it_max):
        I = np.conj(S / (V[..., 1:, :3] - V[..., 1:, 3:4]))          # กระแสเข้าโหลดแต่ละเฟส
        inj = np.concatenate([I, -I.sum(-1, keepdims=True)], -1)       # นิวทรัลรับกระแสกลับ
        Ja = inj[..., :N_A, :][..., ::-1, :].cumsum(-2)[..., ::-1, :]  # กระแสในแต่ละช่วงสาย 1
        Jb = inj[..., N_A:, :][..., ::-1, :].cumsum(-2)[..., ::-1, :]
        itr = Ja[..., 0, :3] + Jb[..., 0, :3] + np.conj(S_MAG_PH / V[..., 0, :3])
        Vn = np.empty_like(V)
        Vn[..., 0, :3] = E - Z_TR * itr
        Vn[..., 0, 3] = 0
        Vn[..., 1:N_A + 1, :] = Vn[..., 0:1, :] - (Ja @ Z_SPAN.T).cumsum(-2)
        Vn[..., N_A + 1:, :] = Vn[..., 0:1, :] - (Jb @ Z_SPAN.T).cumsum(-2)
        err = np.abs(Vn - V).max()
        V = Vn
        if err < tol:
            break
    return V


def vpn_pu(V):
    """แรงดันเฟสเทียบนิวทรัลที่บัส 1..16 เป็น pu ของ 230.94 V"""
    return np.abs(V[..., 1:, :3] - V[..., 1:, 3:4]) / VPH


# ---- เอนจิน pandapower: ยุบสายนิวทรัลเข้าไปในค่า sequence ของสาย
# นิวทรัลต่อดินที่หม้อแปลงจุดเดียว กระแสจึงกลับทางนิวทรัลทั้งหมด
# แรงดันเฟสเทียบนิวทรัล: dV = [Zpp + (Znn - 2 Zpn) 1 1^T] Ip = Z1 (I + 1 1^T) Ip
# ได้ Z1 เท่าเดิมตาม datasheet และ Z0 = 4 x Z1 (ระบบ 3 เฟส 3 สายจะไม่ใช่ค่านี้)
_Zred = Z_KM[:3, :3] + (Z_KM[3, 3] - 2 * Z_KM[0, 3]) * np.ones((3, 3))
Z1_SEQ = _Zred[0, 0] - _Zred[0, 1]
Z0_SEQ = _Zred[0, 0] + 2 * _Zred[0, 1]


def build_net():
    """สร้างโครงข่ายใน pandapower (โหลดต่อเฟส-นิวทรัล ตัวละบัส)"""
    import pandapower as pp
    net = pp.create_empty_network()
    mv = pp.create_bus(net, vn_kv=22.0)
    pp.create_ext_grid(net, mv, vm_pu=1.0, s_sc_max_mva=1e5, rx_max=0.1, x0x_max=1.0, r0x0_max=0.1)
    lv = pp.create_bus(net, vn_kv=0.4, name="LV")
    pp.create_transformer_from_parameters(
        net, mv, lv, sn_mva=TR_KVA / 1000, vn_hv_kv=22.0, vn_lv_kv=0.4,
        vk_percent=TR_VK, vkr_percent=TR_VKR, pfe_kw=PFE_KW, i0_percent=I0_PCT,
        tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_step_percent=TAP_STEP,
        tap_pos=TAP_POS, tap_changer_type="Ratio", vector_group="Dyn",
        vk0_percent=TR_VK, vkr0_percent=TR_VKR, mag0_percent=100, mag0_rx=0, si0_hv_partial=0.9)
    buses, loads = [], []
    for f, n in (("1", N_A), ("2", N_B)):
        prev = lv
        for k in range(n):
            b = pp.create_bus(net, vn_kv=0.4, name=f"{f}-{k + 1}")
            pp.create_line_from_parameters(
                net, prev, b, length_km=SPAN_M / 1000,
                r_ohm_per_km=Z1_SEQ.real, x_ohm_per_km=Z1_SEQ.imag, c_nf_per_km=0, max_i_ka=0.216,
                r0_ohm_per_km=Z0_SEQ.real, x0_ohm_per_km=Z0_SEQ.imag, c0_nf_per_km=0)
            buses.append(b)
            prev = b
    for b in buses:
        loads.append(pp.create_asymmetric_load(net, b, p_a_mw=0, p_b_mw=0, p_c_mw=0,
                                               q_a_mvar=0, q_b_mvar=0, q_c_mvar=0))
    return pp, net, np.array(buses), np.array(loads)


_NET = None


def day_voltages(S_day, every=1):
    """แรงดัน pu ของหนึ่งวัน (T', 16, 3) พร้อมดัชนีเวลาที่ใช้"""
    idx = np.arange(0, S_day.shape[0], every)
    if ENGINE == "fast":
        return vpn_pu(solve(S_day[idx])), idx
    global _NET
    if _NET is None:
        _NET = build_net()
    pp, net, buses, loads = _NET
    out = np.empty((len(idx), N_BUS, 3))
    for j, t in enumerate(idx):
        s = S_day[t] / 1e6
        for c, ph in enumerate("abc"):
            net.asymmetric_load.loc[loads, f"p_{ph}_mw"] = s[:, c].real
            net.asymmetric_load.loc[loads, f"q_{ph}_mvar"] = s[:, c].imag
        pp.runpp_3ph(net)
        r = net.res_bus_3ph
        out[j] = np.c_[r.vm_a_pu[buses], r.vm_b_pu[buses], r.vm_c_pu[buses]]
    return out, idx


# =============================================================== 3) การสุ่ม
def draw(n, n_pv=N_PV, n_ev=N_EV, seed=SEED, phase_mode=PHASE_MODE):
    rng = np.random.default_rng(seed)
    n_pv = int(n_pv)
    has_pv = np.zeros((n, N_HOUSE), bool)
    np.put_along_axis(has_pv, rng.random((n, N_HOUSE)).argsort(1)[:, :n_pv], True, 1)
    ev_house = rng.random((n, N_HOUSE)).argsort(1)[:, :n_ev]            # บ้านละคันเดียว
    ev_arr = np.clip(rng.normal(EV_ARRIVE_MEAN, EV_ARRIVE_SD, (n, n_ev)), 16.5, 22.5)
    ev_dur = rng.uniform(*EV_KWH_RANGE, (n, n_ev)) / EV_KW
    phase = (np.tile(np.arange(N_HOUSE) % 3, (n, 1)) if phase_mode == "rotate"
             else rng.integers(0, 3, (n, N_HOUSE)))
    h_scale = rng.uniform(*HOUSE_SCALE, (n, N_HOUSE))
    h_shift = rng.normal(0, HOUSE_SHIFT_SD, (n, N_HOUSE))
    return dict(has_pv=has_pv, ev_house=ev_house, ev_arr=ev_arr, ev_dur=ev_dur, phase=phase,
                h_scale=h_scale, h_shift=h_shift, n=n)


def inputs(sc, sl):
    """กำลังต่อบัสต่อเฟส (m, T, 16, 3) เป็น VA"""
    hp, ph = sc["has_pv"][sl], sc["phase"][sl]
    m = hp.shape[0]
    oh = (ph[..., None] == np.arange(3)).astype(float)
    n_house = oh.reshape(m, N_BUS, HPB, 3).sum(2)
    pv_kwp = (oh * hp[..., None] * PV_KWP).reshape(m, N_BUS, HPB, 3).sum(2)
    h = hours()
    ev = np.zeros((m, STEPS, N_BUS, 3))
    eh, ea, ed = sc["ev_house"][sl], sc["ev_arr"][sl], sc["ev_dur"][sl]
    rows = np.arange(m)
    for j in range(eh.shape[1]):
        on = (h[None, :] >= ea[:, j:j + 1]) & (h[None, :] < (ea[:, j] + ed[:, j])[:, None])
        ev[rows, :, eh[:, j] // HPB, ph[rows, eh[:, j]]] += on * EV_KW
    pv = clear_sky_pu()
    hs = house_profiles(sc["h_scale"][sl], sc["h_shift"][sl])           # (m, T, บ้าน) kW
    pbus = np.zeros((m, STEPS, N_BUS, 3))                              # รวมบ้านเข้าบัสตามเฟสที่ต่อ
    for q in range(HPB):
        idx = np.arange(N_BUS) * HPB + q
        pbus += hs[:, :, idx][..., None] * oh[:, idx, :][:, None]
    p = pbus + ev - pv_kwp[:, None] * pv[None, :, None, None]
    return (p + 1j * pbus * np.tan(np.arccos(PF))) * 1000


def run(sc, balanced=False, chunk=100, every=None):
    """คืนนาทีที่เกิน/ต่ำ, แรงดันสูงสุด/ต่ำสุดรายเวลา, โอกาสหลุดรายบัส
    balanced=True: เกลี่ยกำลังทุกบัสเท่ากันทุกเฟส = คิดแบบ 3 เฟสสมดุล"""
    every = STEP_EVERY if every is None else every
    dt = DT_MIN * every
    keys = ("over", "under", "vmax_t", "vmin_t", "bus_over", "bus_under")
    out = {k: [] for k in keys}
    for i in range(0, sc["n"], chunk):
        S = inputs(sc, slice(i, min(i + chunk, sc["n"])))
        if balanced:
            S = np.repeat(S.sum(-1, keepdims=True) / 3, 3, -1)
        if ENGINE == "fast":
            idx = np.arange(0, STEPS, every)
            v = vpn_pu(solve(S[:, idx]))                               # (m, T', 16, 3)
        else:
            v = np.stack([day_voltages(S[k], every)[0] for k in range(S.shape[0])])
        vmax_t, vmin_t = v.max((2, 3)), v.min((2, 3))
        out["over"].append((vmax_t > V_HIGH).sum(1) * dt)
        out["under"].append((vmin_t < V_LOW).sum(1) * dt)
        out["vmax_t"].append(vmax_t)
        out["vmin_t"].append(vmin_t)
        out["bus_over"].append((v > V_HIGH).any((1, 3)))
        out["bus_under"].append((v < V_LOW).any((1, 3)))
    return {k: np.concatenate(out[k]) for k in keys}


# =============================================================== 4) รัน
t0 = time.time()
h = hours()
h_mc = h[::STEP_EVERY]                          # แกนเวลาของผล Monte Carlo
print(f"เอนจิน power flow: {ENGINE}, สุ่ม {N_DRAW:,} กรณี, คำนวณทุก {DT_MIN * STEP_EVERY} นาที")
sc = draw(N_DRAW)
r4 = run(sc)
rb = run(sc, balanced=True)
print(f"คำนวณ {N_DRAW:,} กรณี ทั้งแบบ 4 สายและแบบสมดุล: {time.time() - t0:.0f} วินาที")
_hs = house_profiles(sc["h_scale"], sc["h_shift"])
print(f"โหลดบ้าน: พีคต่อหลัง median {np.median(_hs.max(1)):.2f} kW (สูงสุด {HOUSE_PEAK_KW:.0f} kW), "
      f"{np.median(_hs.sum(1) * DT_H):.1f} kWh ต่อวัน, พีครวมทั้งหมู่บ้าน {np.median(_hs.sum(-1).max(1)):.0f} kW")
del _hs
tot = r4["over"] + r4["under"]
ok4, okb = (tot == 0).mean(), ((rb["over"] + rb["under"]) == 0).mean()

if RUN_HOSTING:
    shares = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    evs = np.array([0, 4, 8, 16, 24])
    p_any = np.zeros((len(evs), len(shares)))
    for i, ne in enumerate(evs):
        for j, s in enumerate(shares):
            r = run(draw(N_HOST, int(round(s * N_HOUSE)), int(ne), seed=100 + 10 * i + j))
            p_any[i, j] = ((r["over"] + r["under"]) > 0).mean()
    print(f"ตาราง hosting: {time.time() - t0:.0f} วินาที")

print(f"\nไม่ละเมิดเกณฑ์ตลอดวัน: แบบสมดุล {okb:.1%}, แบบ 4 สาย {ok4:.1%}")
print(f"กรณีที่พบแรงดันสูงกว่า {V_HIGH} pu: สมดุล {(rb['over'] > 0).mean():.1%}, 4 สาย {(r4['over'] > 0).mean():.1%}")
print(f"กรณีที่พบแรงดันต่ำกว่า {V_LOW} pu: สมดุล {(rb['under'] > 0).mean():.1%}, 4 สาย {(r4['under'] > 0).mean():.1%}")
_u = r4["under"][r4["under"] > 0]
if len(_u):
    print(f"เฉพาะกรณีที่ต่ำกว่าเกณฑ์ (4 สาย): มัธยฐาน {np.median(_u):.0f} นาทีต่อวัน, นานที่สุด {_u.max():.0f} นาที")
names = [f"1-{i}" for i in range(1, N_A + 1)] + [f"2-{i}" for i in range(1, N_B + 1)]
print("โอกาสหลุดรายบัส 4 สาย (เกิน/ต่ำ):", ", ".join(
    f"{n} {a:.0%}/{b:.0%}" for n, a, b in zip(names, r4["bus_over"].mean(0), r4["bus_under"].mean(0))))

np.savez_compressed("mc_results.npz", engine=ENGINE, n_draw=N_DRAW,
                    over=r4["over"], under=r4["under"], over_bal=rb["over"],
                    under_bal=rb["under"], vmax_t=r4["vmax_t"], vmin_t=r4["vmin_t"],
                    bus_over=r4["bus_over"], bus_under=r4["bus_under"])

# =============================================================== 5) รูป
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, HOUSE = "#e1e0d9", "#c3c2b7", "#8a8781"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK2,
})
PH = "ABC"


def head(fig, title, sub=None, size=13.5):
    hin = fig.get_size_inches()[1]
    fig.text(0.02, 1 - 0.14 / hin, title, fontsize=size, weight="bold", va="top")
    if not sub:
        return 1 - 0.62 / hin
    fig.text(0.02, 1 - 0.50 / hin, sub, fontsize=10, color=INK2, va="top")
    return 1 - 0.90 / hin


def limit(ax, y, lab=None, x=0.3):
    ax.axhline(y, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=4)
    if lab:
        ax.text(x, y, f" {lab}", fontsize=9, color=INK2, va="bottom")


def hticks(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{x:02d}:00" for x in range(0, 25, 3)])


def show(fig, name):
    fig.savefig(name, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


# ---- รูป 1: เทียบแบบสมดุลกับแบบ 4 สาย
cats = ["สมดุล 3 เฟส", "3 เฟส 4 สาย"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
for ax in (a1, a2):
    ax.grid(axis="x", visible=False)
vals = np.array([okb, ok4]) * 100
a1.bar(range(2), vals, 0.5, color=[MUTED, ORANGE])
for i, v in enumerate(vals):
    a1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, color=INK2)
a1.set_xticks(range(2))
a1.set_xticklabels(cats)
a1.set_ylim(0, 118)
a1.set_title("สัดส่วนกรณีที่ไม่ละเมิดเกณฑ์ตลอด 24 ชั่วโมง", loc="left", fontsize=11)
# แผงขวา: สัดส่วนกรณีที่พบการละเมิด แยกตามชนิด (ค่ามัธยฐานของทั้งชุดเป็นศูนย์เมื่อส่วนใหญ่ไม่ละเมิด)
po = [(rb["over"] > 0).mean() * 100, (r4["over"] > 0).mean() * 100]
pu = [(rb["under"] > 0).mean() * 100, (r4["under"] > 0).mean() * 100]
xx = np.arange(2)
a2.bar(xx - 0.2, po, 0.38, color=ORANGE, label=f"สูงกว่า {V_HIGH:.2f} pu")
a2.bar(xx + 0.2, pu, 0.38, color=BLUE, label=f"ต่ำกว่า {V_LOW:.2f} pu")
for xi, o, u in zip(xx, po, pu):
    a2.text(xi - 0.2, o + 0.8, f"{o:.1f}%", ha="center", fontsize=9.5, color=INK2)
    a2.text(xi + 0.2, u + 0.8, f"{u:.1f}%", ha="center", fontsize=9.5, color=INK2)
_vu = r4["under"][r4["under"] > 0]
if len(_vu):
    a2.text(0.12, max(po + pu) * 0.42,
            f"เฉพาะกรณีที่ละเมิด\nค่ามัธยฐาน {np.median(_vu):.0f} นาทีต่อวัน\nสูงสุด {_vu.max():.0f} นาที",
            ha="left", fontsize=9, color=INK2)
a2.set_xticks(xx)
a2.set_xticklabels(cats)
a2.set_ylim(0, max(po + pu) * 1.45 + 2)
a2.set_title("สัดส่วนกรณีที่พบการละเมิด แยกตามชนิด (%)", loc="left", fontsize=11)
a2.legend(loc="upper left", fontsize=9)
top = head(fig, "เปรียบเทียบแบบจำลองสมดุลกับ 3 เฟส 4 สาย บนชุดตัวอย่างเดียวกัน",
           f"{N_DRAW:,} กรณี, โซลาร์ {N_PV} ราย, EV {N_EV} จุด, ไม่มีระบบกักเก็บ")
fig.tight_layout(rect=[0, 0, 1, top])
show(fig, "mc_compare.png")

# ---- รูป 2: ผังวงจร ตัวอย่าง 3 กรณี
# ทุกเสามีสาย A B C N, บ้านต่อระหว่างเฟสหนึ่งกับ N, วาดบ้านเรียงใต้เฟสที่ต่อ
# เลือก 3 กรณีให้เห็นช่วงของผล: ไม่ละเมิด, เวลาที่ละเมิดใกล้ค่ามัธยฐาน, เวลาที่ละเมิดนานที่สุด
_bad = np.flatnonzero(tot > 0)
if len(_bad):
    _med = _bad[np.argmin(np.abs(tot[_bad] - np.median(tot[_bad])))]
    picks = [int(np.argmin(tot)), int(_med), int(np.argmax(tot))]
    labs = ["ไม่ละเมิดเกณฑ์" if tot[picks[0]] == 0 else f"กรณีที่ละเมิดน้อยที่สุดใน {N_DRAW:,} กรณี",
            "ละเมิดเกณฑ์ เวลาใกล้ค่ามัธยฐานของกรณีที่ละเมิด", "ละเมิดเกณฑ์เป็นเวลานานที่สุด"]
else:
    picks = [0, 1, 2]
    labs = ["ไม่ละเมิดเกณฑ์"] * 3
sub = {k: (v[picks] if isinstance(v, np.ndarray) else v) for k, v in sc.items()}
sub["n"] = 3
S3 = inputs(sub, slice(0, 3))
v3 = np.stack([day_voltages(S3[k], 1)[0] for k in range(3)])          # (3, T, 16, 3) ราย 5 นาที
X1, X2, Y1, Y2 = 3.0 + np.arange(N_A), 3.0 + np.arange(N_B), 0.0, -2.55
SQ, DX = 0.16, 0.27                                                      # ขนาดบ้าน, ระยะคอลัมน์เฟส
fig, axes = plt.subplots(3, 1, figsize=(12, 12.4))
for r, (ax, i) in enumerate(zip(axes, picks)):
    ax.set_xlim(-0.3, 14.6)
    ax.set_ylim(-3.85, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0.25, -1.25), 0.28, fill=False, lw=1.5, color=INK))
    ax.text(0.25, -1.25, "~", ha="center", va="center", fontsize=12)
    ax.add_patch(Circle((1.05, -1.25), 0.22, fill=False, lw=1.5, color=INK))
    ax.add_patch(Circle((1.33, -1.25), 0.22, fill=False, lw=1.5, color=INK))
    ax.plot([0.53, 0.83], [-1.25, -1.25], color=INK, lw=1.5)
    ax.plot([1.55, 2.0], [-1.25, -1.25], color=INK, lw=1.5)
    ax.plot([2.0, 2.0], [Y1 + 0.15, Y2 - 0.15], color=INK, lw=4)
    ax.text(1.2, -1.7, "250 kVA\nDyn11", ha="center", va="top", fontsize=8.5, color=INK2)
    hp, ph, eh = sc["has_pv"][i], sc["phase"][i], sc["ev_house"][i]
    for xs, y, fd, off in ((X1, Y1, "1", 0), (X2, Y2, "2", N_A)):
        ax.plot([2.0, xs[-1]], [y, y], color=INK2, lw=2.2, zorder=1)
        ax.text(xs[-1] + 0.45, y, f"วงจร {fd} ({len(xs) * SPAN_M} m)", va="center", fontsize=9, color=INK2)
        for k, x in enumerate(xs):
            b = off + k
            ax.plot(x, y, "o", color=INK, ms=5.5, zorder=3)
            ax.text(x, y + 0.2, f"{fd}-{k + 1}", ha="center", fontsize=9, color=INK2)
            for p in range(3):
                xp = x + (p - 1) * DX
                ax.text(xp, y - 0.2, PH[p], ha="center", va="center", fontsize=7.5, color=MUTED)
                on_p = [hh for hh in range(b * HPB, (b + 1) * HPB) if ph[hh] == p]
                for j, hh in enumerate(on_p):
                    yq = y - 0.34 - j * (SQ + 0.05)
                    ax.add_patch(Rectangle((xp - SQ / 2, yq - SQ), SQ, SQ, color=HOUSE, lw=0, zorder=2))
                    if hp[hh]:
                        ax.add_patch(Rectangle((xp - SQ / 2, yq), SQ, 0.05, color=YELLOW, lw=0, zorder=2))
                    if hh in eh:
                        ax.add_patch(Circle((xp, yq - SQ / 2), 0.065, color=AQUA, lw=0, zorder=3))
            marks = []
            vmx, vmn = v3[r, :, b].max(0), v3[r, :, b].min(0)
            if vmx.max() > V_HIGH:
                marks.append(("▲", f"{PH[int(vmx.argmax())]} {vmx.max():.3f}", ORANGE))
            if vmn.min() < V_LOW:
                marks.append(("▼", f"{PH[int(vmn.argmin())]} {vmn.min():.3f}", BLUE))
            for j, (sym, val, col) in enumerate(marks):
                ax.text(x + 0.10, y + 0.48 + j * 0.25, sym, fontsize=8.5, color=col, ha="right", va="center")
                ax.text(x + 0.12, y + 0.48 + j * 0.25, val, fontsize=7, color=INK2, ha="left", va="center")
    ax.text(-0.3, 1.25, f"กรณีที่ {i + 1}: {labs[r]}", fontsize=12, weight="bold", va="top")
    ax.text(-0.3, 0.88, f"เวลาที่แรงดันสูงกว่า 1.10 pu {r4['over'][i]} นาที, ต่ำกว่า 0.90 pu {r4['under'][i]} นาที",
            fontsize=10, color=INK2, va="top")
fig.text(0.015, 0.985, f"ตัวอย่างการสุ่ม 3 กรณี จากทั้งหมด {N_DRAW:,} กรณี (แบบจำลอง 3 เฟส 4 สาย)", fontsize=15,
         weight="bold", va="top")
fig.text(0.015, 0.958, "ทุกเสาประกอบด้วยตัวนำ A B C N ผู้ใช้ไฟแต่ละรายต่อระหว่างเฟสหนึ่งกับนิวทรัล วาดเรียงใต้อักษรเฟสที่ต่ออยู่ "
         f"(เสาละ {HPB} ราย คือเฟสละ 1 ราย)", fontsize=9.5, color=INK2, va="top")
# คำอธิบายสัญลักษณ์ วาดของจริงให้ดู
lg = fig.add_axes([0.015, 0.918, 0.97, 0.022])
lg.set_xlim(0, 100)
lg.set_ylim(0, 1)
lg.axis("off")
for x0, kind, text in ((0, "house", "ผู้ใช้ไฟ"), (9, "pv", f"ติดตั้งโซลาร์ {PV_KWP:.0f} kWp"),
                       (29, "ev", f"มีเครื่องอัดประจุ EV {EV_KW} kW"), (48, "up", "แรงดันสูงกว่า 1.10 pu"),
                       (66, "dn", "ต่ำกว่า 0.90 pu (ระบุเฟสและค่าสุดขั้วของวัน)")):
    if kind in ("house", "pv", "ev"):
        lg.add_patch(Rectangle((x0, 0.1), 1.4, 0.65, color=HOUSE, lw=0))
        if kind == "pv":
            lg.add_patch(Rectangle((x0, 0.75), 1.4, 0.2, color=YELLOW, lw=0))
        if kind == "ev":
            lg.plot(x0 + 0.7, 0.42, "o", ms=5, color=AQUA)
    else:
        lg.text(x0 + 0.7, 0.45, "▲" if kind == "up" else "▼", ha="center", va="center", fontsize=10,
                color=ORANGE if kind == "up" else BLUE)
    lg.text(x0 + 2.2, 0.45, text, va="center", fontsize=9.5, color=INK2)
fig.text(0.015, 0.012, "ระบบทดสอบสมมติ: หม้อแปลง 22/0.4 kV 250 kVA vk 4% tap +2.5%, สาย ABC 4x95 mm² "
         "(นิวทรัลขนาดเท่าตัวนำเฟส ต่อลงดินเฉพาะที่หม้อแปลง)", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.01, right=0.99, top=0.915, bottom=0.03, hspace=0.06)
show(fig, "mc_circuit.png")

# ---- รูป 3: แรงดัน 3 เฟสที่ปลายวงจร 1 (กรณีใกล้ค่ามัธยฐานของกรณีที่ละเมิด)
v = v3[1][:, N_A - 1]
fig, ax = plt.subplots(figsize=(9, 4.8))
limit(ax, V_HIGH, f"{V_HIGH} pu", x=18.5)
limit(ax, V_LOW, f"{V_LOW} pu", x=0.3)
for p, col in zip(range(3), (BLUE, ORANGE, AQUA)):
    ax.plot(h, v[:, p], color=col, label=f"เฟส {PH[p]}")
ax.set_ylabel("แรงดันเฟส-นิวทรัล (pu)")
hticks(ax)
ax.set_xlabel("เวลา")
ax.legend(loc="upper left", fontsize=9)
top = head(fig, f"แรงดันเฟส-นิวทรัลรายเฟส ที่เสาปลายวงจร 1 (กรณีที่ {picks[1] + 1})",
           "จุดวัดเดียวกัน แต่ละเฟสได้รับแรงดันต่างกันตามการกระจายของโซลาร์และ EV")
fig.tight_layout(rect=[0, 0, 1, top])
show(fig, "mc_phases.png")

# ---- รูป 4: โอกาสหลุดรายบัส
x = np.r_[np.arange(N_A), np.arange(N_B) + N_A + 1]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.grid(axis="x", visible=False)
ax.bar(x - 0.2, r4["bus_over"].mean(0) * 100, 0.38, color=ORANGE, label="แรงดันสูงกว่า 1.10 pu")
ax.bar(x + 0.2, r4["bus_under"].mean(0) * 100, 0.38, color=BLUE, label="แรงดันต่ำกว่า 0.90 pu")
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n{int(n.split('-')[1]) * SPAN_M} m" for n in names], fontsize=8.5)
ax.set_ylabel(f"ความน่าจะเป็น (% ของ {N_DRAW:,} กรณี)")
_ytop = max(15, max(r4["bus_over"].mean(0).max(), r4["bus_under"].mean(0).max()) * 100 * 1.55)
ax.set_ylim(0, _ytop)
ax.text((N_A - 1) / 2, _ytop * 0.95, f"วงจร 1 ({N_A * SPAN_M} m)", ha="center", fontsize=10.5, color=INK2)
ax.text(N_A + 1 + (N_B - 1) / 2, _ytop * 0.95, f"วงจร 2 ({N_B * SPAN_M} m)", ha="center", fontsize=10.5, color=INK2)
ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88), fontsize=9)
top = head(fig, "ความน่าจะเป็นที่แรงดันละเมิดเกณฑ์ แยกตามเสา (นับเมื่อมีเฟสใดเฟสหนึ่งละเมิด)")
fig.tight_layout(rect=[0, 0, 1, top])
show(fig, "mc_bus.png")

# ---- รูป 5: แรงดันตลอดวัน
fig, ax = plt.subplots(figsize=(9, 5.4))
for arr, col, lab in ((r4["vmax_t"], ORANGE, "แรงดันสูงสุดในวงจร"),
                      (r4["vmin_t"], BLUE, "แรงดันต่ำสุดในวงจร")):
    f5, f25, f50, f75, f95 = np.percentile(arr, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(h_mc, f5, f95, color=col, alpha=0.12, lw=0)
    ax.fill_between(h_mc, f25, f75, color=col, alpha=0.28, lw=0)
    ax.plot(h_mc, f50, color=col, label=f"ค่ามัธยฐานของ{lab}")
limit(ax, V_HIGH, f"{V_HIGH} pu", x=0.4)                  # วาดทีหลังให้อยู่เหนือแถบ
limit(ax, V_LOW, f"{V_LOW} pu", x=0.4)
ax.set_ylabel("แรงดันเฟส-นิวทรัล (pu)")
hticks(ax)
ax.set_xlabel("เวลา")
ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=9)
top = head(fig, f"การกระจายตัวของแรงดันสูงสุดและต่ำสุดในวงจร ตลอด 24 ชั่วโมง ({N_DRAW:,} กรณี)",
           "เส้นทึบคือค่ามัธยฐาน แถบเข้มคือเปอร์เซ็นไทล์ที่ 25 ถึง 75 แถบอ่อนคือ 5 ถึง 95")
fig.tight_layout(rect=[0, 0, 1, top])
show(fig, "mc_fan.png")

# ---- รูป 6: hosting capacity
if RUN_HOSTING:
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
    P = p_any * 100
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.grid(False)
    im = ax.imshow(P, cmap=ListedColormap(ramp), vmin=0, vmax=100, origin="lower", aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                    color="#ffffff" if P[i, j] > 55 else INK)
    ax.set_xticks(range(len(shares)))
    ax.set_xticklabels([f"{int(s * N_HOUSE)} หลัง\n({s * 100:.0f}%)" for s in shares])
    ax.set_yticks(range(len(evs)))
    ax.set_yticklabels([f"{e} คัน" for e in evs])
    ax.set_xlabel(f"บ้านที่ติดโซลาร์ {PV_KWP:.0f} kWp (จาก {N_HOUSE} หลัง)")
    ax.set_ylabel("จำนวน EV")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("% ที่มีบ้านหลุดเกณฑ์", color=INK2)
    top = head(fig, "Hosting capacity 3 เฟส 4 สาย", f"แต่ละช่องสุ่ม {N_HOST} กรณี")
    fig.tight_layout(rect=[0, 0, 1, top])
    show(fig, "mc_hosting.png")

print("\nบันทึกรูป: mc_compare.png, mc_circuit.png, mc_phases.png, mc_bus.png, mc_fan.png"
      + (", mc_hosting.png" if RUN_HOSTING else ""))
