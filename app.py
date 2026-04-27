import streamlit as st
import rawpy
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pillow_lut
import exifread
from sklearn.cluster import KMeans
import io
import os
import gc

# --- CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(layout="wide", page_title="RAW Studio Pro", page_icon="📷")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stDownloadButton > button {
        width: 100% !important;
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 8px !important;
        height: 3.5em !important;
        font-weight: bold !important;
    }
    .img-card {
        border: 1px solid #333;
        border-radius: 15px;
        padding: 20px;
        background-color: #1a1c23;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE PROCESAMIENTO OPTIMIZADAS ---

def get_optimized_size(w, h, short_edge=1080):
    """Calcula las dimensiones para que el borde corto sea de 1080px."""
    if w < h: # Retrato
        new_w = short_edge
        new_h = int(h * (short_edge / w))
    else: # Paisaje
        new_h = short_edge
        new_w = int(w * (short_edge / h))
    return new_w, new_h

@st.cache_data(show_spinner=False)
def get_processed_base(file_bytes, auto_bright=False):
    """Revelado base optimizado para evitar colapsos de memoria."""
    try:
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            try:
                # Intentar desempaquetado normal
                raw.unpack()
            except Exception:
                # FALLBACK: Si es un NEF de cámara nueva (Z50II) y falla, intentamos extraer la imagen embebida
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        img = Image.open(io.BytesIO(thumb.data))
                        return np.array(img)
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        return thumb.data
                except:
                    return "Error: Formato RAW no soportado por el motor LibRaw."

            # Usamos half_size=True para la previsualización si el archivo es muy grande
            # Esto ahorra un 75% de memoria durante la edición
            rgb = raw.postprocess(
                use_camera_wb=True, 
                no_auto_bright=not auto_bright, 
                output_color=rawpy.ColorSpace.sRGB,
                half_size=True, # Optimización clave para evitar OOM
                bright=1.0
            )
            return rgb
    except Exception as e:
        return str(e)

def apply_adjustments(rgb_array, params, lut_file=None):
    """Aplica ajustes sobre la imagen optimizada."""
    # Convertimos a float32 pero de forma controlada
    img_data = rgb_array.astype(np.float32) / 255.0
    
    # Balance de Blancos
    if params['temp'] != 0:
        t = params['temp'] / 10.0
        img_data[:, :, 0] *= (1.0 + t)
        img_data[:, :, 2] *= (1.0 - t)
        np.clip(img_data, 0, 1, out=img_data)

    # Sombras y Luces (Optimizado para no crear múltiples copias en RAM)
    if params['shadows'] != 0:
        gamma_s = 1.0 - (params['shadows'] / 4.0)
        mask = img_data < 0.5
        img_data[mask] = np.power(img_data[mask] * 2, gamma_s) / 2
        
    if params['highlights'] != 0:
        gamma_h = 1.0 + (params['highlights'] / 4.0)
        mask = img_data >= 0.5
        img_data[mask] = 1.0 - (np.power((1.0 - img_data[mask]) * 2, gamma_h) / 2)

    img = Image.fromarray((np.clip(img_data, 0, 1) * 255).astype(np.uint8))
    
    # Liberar memoria de array float
    del img_data
    
    # Ajustes PIL (Más eficientes en memoria)
    if params['exposure'] != 0:
        img = ImageEnhance.Brightness(img).enhance(1.0 + (params['exposure'] / 5.0))
    if params['contrast'] != 1.0:
        img = ImageEnhance.Contrast(img).enhance(params['contrast'])
    if params['saturation'] != 1.0:
        img = ImageEnhance.Color(img).enhance(params['saturation'])
    
    if params['clarity'] > 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=int(params['clarity'] * 100)))
    
    if lut_file:
        try:
            lut = pillow_lut.load_cube_file(lut_file)
            img = img.filter(lut)
        except: pass
            
    return img

def create_social_export(img, file_bytes, palette):
    """Genera la pieza final de 1080px de borde corto."""
    # Redimensionar al tamaño final solicitado
    target_w, target_h = get_optimized_size(img.width, img.height, 1080)
    img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    # Datos EXIF
    try:
        tags = exifread.process_file(io.BytesIO(file_bytes), details=False)
        exif_text = f"ISO {tags.get('EXIF ISOSpeedRatings', 'N/A')} | f/{tags.get('EXIF FNumber', 'N/A')} | {tags.get('EXIF ExposureTime', 'N/A')}s"
    except:
        exif_text = "Metadata no disponible"

    # Lienzo
    footer_h = int(target_h * 0.15)
    canvas = Image.new('RGB', (target_w, target_h + footer_h), (15, 15, 15))
    canvas.paste(img_resized, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Paleta
    pw = target_w // 10
    start_x = (target_w - (pw * 5)) // 2
    for i, color in enumerate(palette):
        x0 = start_x + (i * pw)
        draw.rectangle([x0, target_h + 20, x0 + pw - 10, target_h + footer_h - 40], fill=color)
    
    try:
        draw.text((target_w//2, target_h + footer_h - 22), exif_text, fill=(130, 130, 130), anchor="ms")
    except: pass
    
    return canvas

def get_palette(img):
    img_small = img.resize((50, 50))
    ar = np.asarray(img_small).reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, n_init=5).fit(ar)
    return [tuple(c) for c in kmeans.cluster_centers_.astype(int)]

# --- INTERFAZ ---

st.title("📷 RAW Studio: Engine Optimizado")

with st.sidebar:
    st.header("⚡ Revelado")
    auto_mode = st.toggle("🚀 Auto-Brillo AI", value=False)
    lut_upload = st.file_uploader("Estilo (.CUBE)", type=['cube'])
    
    with st.expander("Luz y Color", expanded=True):
        exp = st.slider("Exposición", -5.0, 5.0, 0.0, 0.1)
        con = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
        sat = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1)
        temp = st.slider("Temperatura", -2.0, 2.0, 0.0, 0.1)

    with st.expander("Detalles Avanzados"):
        sha = st.slider("Sombras", -1.0, 1.0, 0.0, 0.1)
        hig = st.slider("Altas Luces", -1.0, 1.0, 0.0, 0.1)
        cla = st.slider("Claridad", 0.0, 2.0, 0.0, 0.1)

    if st.button("Vaciar Memoria Caché"):
        st.cache_data.clear()
        st.rerun()

params = {
    'exposure': exp, 'contrast': con, 'saturation': sat, 
    'temp': temp, 'shadows': sha, 'highlights': hig, 'clarity': cla
}

uploaded_files = st.file_uploader("Sube tus fotos (RAW/DNG)", type=['nef', 'cr2', 'arw', 'dng'], accept_multiple_files=True)

if uploaded_files:
    st.divider()
    cols = st.columns(2)
    
    for idx, file in enumerate(uploaded_files[:10]):
        data = file.getvalue()
        base = get_processed_base(data, auto_bright=auto_mode)
        
        if isinstance(base, np.ndarray):
            # Procesar
            img_edited = apply_adjustments(base, params, lut_upload)
            palette = get_palette(img_edited)
            
            with cols[idx % 2]:
                st.markdown('<div class="img-card">', unsafe_allow_html=True)
                st.image(img_edited, caption=file.name, use_container_width=True)
                
                # Generar exportación Social
                final_jpg = create_social_export(img_edited, data, palette)
                buf = io.BytesIO()
                final_social_img = final_jpg.save(buf, format="JPEG", quality=92)
                
                st.download_button(
                    label="💾 DESCARGAR JPG (1080px)",
                    data=buf.getvalue(),
                    file_name=f"RAW_PRO_{file.name}.jpg",
                    mime="image/jpeg",
                    key=f"dl_{idx}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Error en {file.name}: {base}")
            if ".NEF" in file.name.upper():
                st.warning("Cámara nueva detectada. Se ha intentado extraer el JPG embebido pero el formato es propietario.")
        
        gc.collect()
else:
    st.info("Sube tus archivos para comenzar. El motor se ha optimizado para evitar cierres por falta de memoria.")
