import streamlit as st
import rawpy
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pillow_lut
import exifread
from sklearn.cluster import KMeans
import io
import os
import gc

# Configuración de página para modo ancho
st.set_page_config(layout="wide", page_title="RAW Batch Pro", page_icon="📷")

# Estilos CSS para mejorar la interfaz
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; }
    </style>
    """, unsafe_allow_html=True)

def get_exif_data(file_bytes):
    """Extrae metadatos técnicos de la imagen."""
    try:
        file_bytes.seek(0)
        tags = exifread.process_file(file_bytes, details=False)
        return {
            "ISO": tags.get('EXIF ISOSpeedRatings', 'N/A'),
            "Apertura": tags.get('EXIF FNumber', 'N/A'),
            "Velocidad": tags.get('EXIF ExposureTime', 'N/A'),
            "Lente": tags.get('EXIF LensModel', 'N/A')
        }
    except:
        return {"ISO": "N/A", "Apertura": "N/A", "Velocidad": "N/A", "Lente": "N/A"}

def get_palette(img, n_colors=5):
    """Genera la paleta de colores usando K-Means."""
    img_small = img.resize((100, 100))
    ar = np.asarray(img_small)
    shape = ar.shape
    ar = ar.reshape(np.prod(shape[:2]), shape[2])
    kmeans = KMeans(n_clusters=n_colors, n_init=10).fit(ar)
    return [tuple(c) for c in kmeans.cluster_centers_.astype(int)]

def process_raw(file_bytes, params, auto=False):
    """Revelado del archivo RAW con rawpy."""
    file_bytes.seek(0)
    with rawpy.imread(file_bytes) as raw:
        postprocess_params = {
            "use_camera_wb": True,
            "no_auto_bright": not auto,
            "bright": 1.0 + (params['exposure'] / 5.0) if not auto else 1.0,
            "output_color": rawpy.ColorSpace.sRGB,
        }
        rgb = raw.postprocess(**postprocess_params)
        img = Image.fromarray(rgb)
        
        # Ajustes de saturación
        if params['saturation'] != 1.0:
            from PIL import ImageEnhance
            img = ImageEnhance.Color(img).enhance(params['saturation'])
            
        # Claridad mediante máscara de enfoque
        if params['clarity'] > 0:
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(params['clarity'] * 100)))
            
        return img

def apply_lut(img, lut_file):
    """Aplica un archivo .CUBE a la imagen."""
    if lut_file:
        lut = pillow_lut.load_cube_file(lut_file)
        return img.filter(lut)
    return img

def create_social_export(img, exif, palette):
    """Diseño final con ficha técnica y paleta."""
    w, h = img.size
    footer_h = int(h * 0.15)
    full_img = Image.new('RGB', (w, h + footer_h), (20, 20, 20))
    full_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(full_img)
    
    # Dibujar bloques de color
    block_w = w // 10
    start_x = (w - (block_w * len(palette))) // 2
    for i, color in enumerate(palette):
        x0 = start_x + (i * block_w)
        draw.rectangle([x0, h + 20, x0 + block_w - 10, h + footer_h - 40], fill=color)
        
    # Texto de datos EXIF
    text = f"ISO {exif['ISO']}  |  f/{exif['Apertura']}  |  {exif['Velocidad']}s  |  {exif['Lente']}"
    try:
        draw.text((w//2, h + footer_h - 25), text, fill=(200, 200, 200), anchor="ms")
    except:
        pass # Fallback si no hay fuentes instaladas
        
    return full_img

st.title("📷 RAW Batch Pro & LUT Studio")

if 'global_params' not in st.session_state:
    st.session_state.global_params = {'exposure': 0.0, 'clarity': 0.0, 'saturation': 1.0}

with st.sidebar:
    st.header("🎛️ Controles")
    st.session_state.global_params['exposure'] = st.slider("Exposición", -5.0, 5.0, 0.0, 0.1)
    st.session_state.global_params['clarity'] = st.slider("Claridad", 0.0, 2.0, 0.0, 0.1)
    st.session_state.global_params['saturation'] = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1)
    auto_mode = st.checkbox("Auto-Brillo")
    lut_upload = st.file_uploader("Cargar LUT (.CUBE)", type=['cube'])

uploaded_files = st.file_uploader("Sube tus fotos RAW (máx 10)", type=['nef', 'cr2', 'arw', 'dng', 'orf'], accept_multiple_files=True)

if uploaded_files:
    files_to_process = uploaded_files[:10]
    cols = st.columns(2)
    
    for idx, file in enumerate(files_to_process):
        with st.spinner(f"Procesando {file.name}..."):
            exif = get_exif_data(file)
            img = process_raw(file, st.session_state.global_params, auto=auto_mode)
            
            if lut_upload:
                img = apply_lut(img, lut_upload)
            
            # Generar previsualización optimizada
            preview = img.copy()
            preview.thumbnail((800, 800))
            palette = get_palette(preview)
            
            with cols[idx % 2]:
                st.image(preview, caption=file.name, use_container_width=True)
                
                # Botón de descarga
                final_social = create_social_export(img, exif, palette)
                buf = io.BytesIO()
                final_social.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label=f"⬇️ Descargar {file.name}",
                    data=buf.getvalue(),
                    file_name=f"PRO_{os.path.splitext(file.name)[0]}.jpg",
                    mime="image/jpeg",
                    key=idx
                )
        gc.collect() # Limpiar memoria tras cada foto