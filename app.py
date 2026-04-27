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
        height: 3em !important;
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

# --- FUNCIONES DE PROCESAMIENTO ---

def resize_for_social(img, short_edge=1080):
    """Redimensiona la imagen para que el borde más corto sea de 1080px."""
    w, h = img.size
    if w < h: # Retrato o cuadrado
        new_w = short_edge
        new_h = int(h * (short_edge / w))
    else: # Paisaje
        new_h = short_edge
        new_w = int(w * (short_edge / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

@st.cache_data(show_spinner=False)
def get_base_raw(file_bytes, auto_bright=False):
    """Revelado base del RAW. Cacheado para velocidad."""
    try:
        # Usamos BytesIO para que rawpy pueda leer los bytes como un archivo
        with rawpy.imread(io.BytesIO(file_bytes)) as raw:
            raw.unpack()
            rgb = raw.postprocess(
                use_camera_wb=True, 
                no_auto_bright=not auto_bright, 
                output_color=rawpy.ColorSpace.sRGB,
                user_flip=0,
                bright=1.0
            )
            return rgb
    except Exception as e:
        # Retornamos el error para poder mostrarlo en la UI si es necesario
        return e

def apply_adjustments(rgb_array, params, lut_file=None):
    """Motor de ajustes avanzados."""
    # Procesamiento con Numpy para Sombras/Altas Luces
    img_data = rgb_array.astype(np.float32) / 255.0
    
    # Balance de Blancos (Temperatura)
    if params['temp'] != 0:
        t = params['temp'] / 10.0
        img_data[:, :, 0] *= (1.0 + t)
        img_data[:, :, 2] *= (1.0 - t)
        img_data = np.clip(img_data, 0, 1)

    # Sombras y Altas Luces
    if params['shadows'] != 0:
        gamma_s = 1.0 - (params['shadows'] / 4.0)
        img_data = np.where(img_data < 0.5, np.power(img_data * 2, gamma_s) / 2, img_data)
    if params['highlights'] != 0:
        gamma_h = 1.0 + (params['highlights'] / 4.0)
        img_data = np.where(img_data >= 0.5, 1.0 - (np.power((1.0 - img_data) * 2, gamma_h) / 2), img_data)

    img = Image.fromarray((np.clip(img_data, 0, 1) * 255).astype(np.uint8))
    
    # Ajustes PIL
    img = ImageEnhance.Brightness(img).enhance(1.0 + (params['exposure'] / 5.0))
    img = ImageEnhance.Contrast(img).enhance(params['contrast'])
    img = ImageEnhance.Color(img).enhance(params['saturation'])
    
    if params['clarity'] > 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=int(params['clarity'] * 100)))
    
    if lut_file:
        try:
            lut = pillow_lut.load_cube_file(lut_file)
            img = img.filter(lut)
        except: pass
            
    return img

def create_social_frame(img, file_bytes, palette):
    """Añade la ficha técnica y paleta al JPG final de 1080px (borde corto)."""
    # 1. Redimensionar imagen principal
    img_resized = resize_for_social(img)
    w, h = img_resized.size
    
    # 2. Extraer EXIF
    try:
        tags = exifread.process_file(io.BytesIO(file_bytes), details=False)
        iso = tags.get('EXIF ISOSpeedRatings', 'N/A')
        f_stop = tags.get('EXIF FNumber', 'N/A')
        shutter = tags.get('EXIF ExposureTime', 'N/A')
        exif_text = f"ISO {iso} | f/{f_stop} | {shutter}s"
    except:
        exif_text = "Datos EXIF no disponibles"

    # 3. Crear lienzo con franja inferior
    footer_h = int(h * 0.15)
    canvas = Image.new('RGB', (w, h + footer_h), (15, 15, 15))
    canvas.paste(img_resized, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Dibujar Paleta
    pw = w // 10
    start_x = (w - (pw * len(palette))) // 2
    for i, color in enumerate(palette):
        x0 = start_x + (i * pw)
        draw.rectangle([x0, h + 20, x0 + pw - 10, h + footer_h - 45], fill=color)
    
    # Texto EXIF
    try:
        draw.text((w//2, h + footer_h - 25), exif_text, fill=(150, 150, 150), anchor="ms")
    except: pass
    
    return canvas

def get_palette(img):
    """Extrae 5 colores para el diseño."""
    img_small = img.resize((50, 50))
    ar = np.asarray(img_small).reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, n_init=5).fit(ar)
    return [tuple(c) for c in kmeans.cluster_centers_.astype(int)]

# --- INTERFAZ DE USUARIO ---

st.title("📷 RAW Studio: Social Media Ready")

with st.sidebar:
    st.header("⚡ Ajustes Rápidos")
    auto_mode = st.toggle("🚀 Optimizar Brillo Automático", value=False)
    lut_upload = st.file_uploader("Cargar LUT (.CUBE)", type=['cube'])
    
    with st.expander("Exposición y Color", expanded=True):
        exp = st.slider("Exposición", -5.0, 5.0, 0.0, 0.1)
        con = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
        sat = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1)
        temp = st.slider("Temperatura", -2.0, 2.0, 0.0, 0.1)

    with st.expander("Sombras y Luces"):
        sha = st.slider("Sombras", -1.0, 1.0, 0.0, 0.1)
        hig = st.slider("Altas Luces", -1.0, 1.0, 0.0, 0.1)
        cla = st.slider("Claridad", 0.0, 2.0, 0.0, 0.1)

    if st.button("Limpiar Memoria"):
        st.cache_data.clear()
        st.rerun()

params = {
    'exposure': exp, 'contrast': con, 'saturation': sat, 
    'temp': temp, 'shadows': sha, 'highlights': hig, 'clarity': cla
}

uploaded_files = st.file_uploader("Sube tus archivos RAW", type=['nef', 'cr2', 'arw', 'dng', 'orf'], accept_multiple_files=True)

if uploaded_files:
    st.divider()
    cols = st.columns(2)
    
    for idx, file in enumerate(uploaded_files[:10]):
        # IMPORTANTE: Usamos getvalue() para obtener los bytes sin perder el puntero en reruns
        file_data = file.getvalue()
        
        # Intentamos revelar el RAW
        result = get_base_raw(file_data, auto_bright=auto_mode)
        
        if isinstance(result, np.ndarray):
            # Procesar imagen con sliders
            final_img = apply_adjustments(result, params, lut_upload)
            
            # Crear previsualización liviana para la web
            preview_img = final_img.copy()
            preview_img.thumbnail((1000, 1000))
            palette = get_palette(preview_img)
            
            with cols[idx % 2]:
                st.markdown(f'<div class="img-card">', unsafe_allow_html=True)
                
                # --- VISTA PREVIA ---
                st.image(preview_img, caption=f"Editando: {file.name}", use_container_width=True)
                
                # --- PROCESAR DESCARGA (1080px short edge) ---
                social_jpg = create_social_frame(final_img, file_data, palette)
                buf = io.BytesIO()
                social_jpg.save(buf, format="JPEG", quality=95, subsampling=0)
                
                # --- BOTÓN DE DESCARGA ---
                st.download_button(
                    label=f"💾 DESCARGAR JPG (1080px)",
                    data=buf.getvalue(),
                    file_name=f"SOCIAL_{os.path.splitext(file.name)[0]}.jpg",
                    mime="image/jpeg",
                    key=f"dl_{idx}"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Si 'result' no es un array, es el error capturado
            st.error(f"Error en {file.name}: {result}")
            st.info("Asegúrate de que el archivo no esté corrupto y sea compatible con LibRaw.")
        
        gc.collect()
else:
    st.info("👋 Sube tus fotos RAW para ver la vista previa y aplicar los ajustes.")
