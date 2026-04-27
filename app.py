import streamlit as st
import rawpy
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import pillow_lut
import exifread
from sklearn.cluster import KMeans
import io
import os
import gc

# Configuración de página
st.set_page_config(layout="wide", page_title="RAW Batch Pro v2", page_icon="📷")

# Estilos CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #444; }
    .stButton>button:hover { border-color: #ff4b4b; color: #ff4b4b; }
    .img-container { border: 1px solid #333; border-radius: 12px; padding: 15px; margin-bottom: 25px; background-color: #1a1c23; }
    .sidebar .sidebar-content { background-color: #111; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_base_raw(file_bytes, auto_bright=False):
    """Revelado base del RAW optimizado."""
    try:
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            raw.unpack()
            # Revelado balanceado
            rgb = raw.postprocess(
                use_camera_wb=True, 
                no_auto_bright=not auto_bright, 
                output_color=rawpy.ColorSpace.sRGB,
                bright=1.0,
                user_flip=0
            )
            return rgb
    except Exception as e:
        return None

def get_exif_data(file_bytes):
    try:
        tags = exifread.process_file(io.BytesIO(file_bytes), details=False)
        return {
            "ISO": tags.get('EXIF ISOSpeedRatings', 'N/A'),
            "Apertura": tags.get('EXIF FNumber', 'N/A'),
            "Velocidad": tags.get('EXIF ExposureTime', 'N/A'),
            "Lente": tags.get('EXIF LensModel', 'N/A')
        }
    except:
        return {"ISO": "N/A", "Apertura": "N/A", "Velocidad": "N/A", "Lente": "N/A"}

def apply_advanced_adjustments(rgb_array, params, lut_file=None):
    """Aplica el motor de revelado avanzado."""
    # Convertir a float para cálculos precisos
    img_data = rgb_array.astype(np.float32) / 255.0
    
    # 1. Temperatura de Color (Warmth/Coolness)
    # Ajustamos canales Rojo y Azul de forma inversa
    if params['temp'] != 0:
        t = params['temp'] / 10.0
        img_data[:, :, 0] *= (1.0 + t) # Red
        img_data[:, :, 2] *= (1.0 - t) # Blue
        img_data = np.clip(img_data, 0, 1)

    # 2. Sombras y Altas Luces (Shadows & Highlights)
    # Usamos una curva simple: y = x^(gamma)
    if params['shadows'] != 0:
        # Shadows afecta al rango bajo. Gamma < 1 aclara.
        gamma_s = 1.0 - (params['shadows'] / 4.0)
        img_data = np.where(img_data < 0.5, np.power(img_data * 2, gamma_s) / 2, img_data)
        
    if params['highlights'] != 0:
        # Highlights afecta al rango alto. Gamma > 1 oscurece.
        gamma_h = 1.0 + (params['highlights'] / 4.0)
        img_data = np.where(img_data >= 0.5, 1.0 - (np.power((1.0 - img_data) * 2, gamma_h) / 2), img_data)

    # Reconversión a PIL para filtros estándar
    img = Image.fromarray((np.clip(img_data, 0, 1) * 255).astype(np.uint8))
    
    # 3. Exposición
    if params['exposure'] != 0:
        img = ImageEnhance.Brightness(img).enhance(1.0 + (params['exposure'] / 5.0))
        
    # 4. Contraste
    if params['contrast'] != 1.0:
        img = ImageEnhance.Contrast(img).enhance(params['contrast'])
        
    # 5. Saturación
    if params['saturation'] != 1.0:
        img = ImageEnhance.Color(img).enhance(params['saturation'])
        
    # 6. Claridad vs Nitidez
    if params['clarity'] > 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=int(params['clarity'] * 100)))
    if params['sharpness'] > 1.0:
        img = ImageEnhance.Sharpness(img).enhance(params['sharpness'])
        
    # 7. Viñeta
    if params['vignette'] > 0:
        w, h = img.size
        # Crear máscara de viñeta
        mask = Image.new('L', (w, h), 255)
        draw = ImageDraw.Draw(mask)
        # Dibujar gradiente elíptico
        v_strength = params['vignette']
        for i in range(int(min(w, h) * 0.6)):
            alpha = int(255 * (i / (min(w, h) * 0.6)) ** v_strength)
            draw.ellipse([i, i, w-i, h-i], outline=alpha)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=w/10))
        black = Image.new('RGB', (w, h), (0, 0, 0))
        img = Image.composite(img, black, mask)

    # 8. LUT final
    if lut_file:
        try:
            lut = pillow_lut.load_cube_file(lut_file)
            img = img.filter(lut)
        except:
            pass
            
    return img

def get_palette(img, n_colors=5):
    img_small = img.resize((100, 100))
    ar = np.asarray(img_small)
    shape = ar.shape
    ar = ar.reshape(np.prod(shape[:2]), shape[2])
    kmeans = KMeans(n_clusters=n_colors, n_init=10).fit(ar)
    return [tuple(c) for c in kmeans.cluster_centers_.astype(int)]

def create_social_export(img, exif, palette):
    w, h = img.size
    footer_h = int(h * 0.15)
    full_img = Image.new('RGB', (w, h + footer_h), (15, 15, 15))
    full_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(full_img)
    block_w = w // 10
    start_x = (w - (block_w * len(palette))) // 2
    for i, color in enumerate(palette):
        x0 = start_x + (i * block_w)
        draw.rectangle([x0, h + 20, x0 + block_w - 10, h + footer_h - 40], fill=color)
    text = f"ISO {exif['ISO']}  |  f/{exif['Apertura']}  |  {exif['Velocidad']}s  |  {exif['Lente']}"
    try: draw.text((w//2, h + footer_h - 25), text, fill=(180, 180, 180), anchor="ms")
    except: pass 
    return full_img

# --- INTERFAZ ---
st.title("📷 RAW Studio Pro: Revelado Avanzado")

with st.sidebar:
    st.header("✨ Revelado")
    with st.expander("Básico", expanded=True):
        exp = st.slider("Exposición", -5.0, 5.0, 0.0, 0.1)
        con = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
        temp = st.slider("Temperatura (Frío/Cálido)", -2.0, 2.0, 0.0, 0.1)
        sat = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1)

    with st.expander("Rango Dinámico"):
        sha = st.slider("Sombras", -1.0, 1.0, 0.0, 0.1, help="Recuperar detalle en negros")
        hig = st.slider("Altas Luces", -1.0, 1.0, 0.0, 0.1, help="Recuperar detalle en blancos")

    with st.expander("Detalle y Estilo"):
        cla = st.slider("Claridad", 0.0, 2.0, 0.0, 0.1)
        sha_p = st.slider("Nitidez", 1.0, 3.0, 1.0, 0.1)
        vig = st.slider("Viñeta", 0.0, 3.0, 0.0, 0.1)
    
    st.divider()
    auto_mode = st.checkbox("Optimización de Brillo AI")
    lut_upload = st.file_uploader("Estilo Cine (.CUBE)", type=['cube'])
    
    if st.button("Limpiar Memoria"):
        st.cache_data.clear()
        st.rerun()

params = {
    'exposure': exp, 'contrast': con, 'saturation': sat, 
    'temp': temp, 'shadows': sha, 'highlights': hig,
    'clarity': cla, 'sharpness': sha_p, 'vignette': vig
}

uploaded_files = st.file_uploader("Carga tus archivos RAW", type=['nef', 'cr2', 'arw', 'dng', 'orf'], accept_multiple_files=True)

if uploaded_files:
    n_files = len(uploaded_files[:10])
    cols = st.columns(2)
    
    for idx, file in enumerate(uploaded_files[:10]):
        file_bytes = file.read()
        base_rgb = get_base_raw(file_bytes, auto_bright=auto_mode)
        
        if base_rgb is not None:
            exif = get_exif_data(file_bytes)
            final_img = apply_advanced_adjustments(base_rgb, params, lut_upload)
            
            # Previsualización rápida
            preview = final_img.copy()
            preview.thumbnail((900, 900))
            palette = get_palette(preview)
            
            with cols[idx % 2]:
                st.markdown(f'<div class="img-container">', unsafe_allow_html=True)
                st.image(preview, caption=file.name, use_container_width=True)
                
                final_social = create_social_export(final_img, exif, palette)
                buf = io.BytesIO()
                final_social.save(buf, format="JPEG", quality=95)
                
                st.download_button(
                    label=f"⬇️ Exportar {file.name}",
                    data=buf.getvalue(),
                    file_name=f"PRO_{os.path.splitext(file.name)[0]}.jpg",
                    mime="image/jpeg",
                    key=f"btn_{idx}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        gc.collect()
else:
    st.info("Sube una foto para activar el panel de revelado avanzado.")
