import os
import streamlit as st
from PIL import Image

# Ensure numpy is available
try:
    import numpy as np
except ImportError:
    st.error("NumPy no está disponible. Por favor verifica tu archivo requirements.txt")
    st.stop()

try:
    import torch
    # Apply PyTorch fix for model loading
    from src.pytorch_fix import allow_model_loading
    from torchvision import transforms
except ImportError as e:
    st.error(f"Falló la importación de PyTorch: {e}")
    st.stop()

from src.model import MyModel, load_model
from src.utils import predict
from src.segmentation import TumorSegmentor
from src.fallback_segmentation import create_fallback_visualization
from download_models import download_models, check_models_exist

# Page config
st.set_page_config(
    page_title="Clasificador IA de Tumores Cerebrales",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for model download
if 'models_downloaded' not in st.session_state:
    st.session_state.models_downloaded = False

# Load the trained model with error handling
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
except Exception as e:
    device = "cpu"
    print(f"Defaulting to CPU due to device error: {e}")

model_path = os.path.join("models", "model_38")

# Check if models exist
models_available, missing_models = check_models_exist()

# Show model status in sidebar
with st.sidebar:
    st.subheader("🤖 Estado de los Modelos IA")
    
    if models_available:
        st.success("✅ ¡Todos los modelos IA están listos!")
        st.session_state.models_downloaded = True
    else:
        st.warning(f"❌ Modelos faltantes: {', '.join(missing_models)}")
        
        # Add download button
        if st.button("📥 Descargar Modelos IA", type="primary", help="Descargar modelos desde almacenamiento en la nube"):
            with st.spinner("Descargando modelos..."):
                if download_models():
                    st.session_state.models_downloaded = True
                    st.rerun()  # Refresh to load models
                else:
                    st.error("No se pudieron descargar algunos modelos. Por favor verifica tu conexión a internet e intenta nuevamente.")

# Load models if available
if models_available or st.session_state.models_downloaded:
    try:
        model = load_model(model_path, device)
    except Exception as e:
        st.error(f"Error al cargar el modelo de clasificación: {e}")
        models_available = False
        model = None
else:
    st.info("""
    🚀 **¡Bienvenido al Clasificador IA de Tumores Cerebrales!**
    
    Esta aplicación ofrece tres funcionalidades impulsadas por IA:
    - 🔍 **Clasificación**: Detectar y clasificar tumores cerebrales
    - 📍 **Detección**: Localizar tumores con cuadros delimitadores  
    - 🎯 **Segmentación**: Mapeo preciso del área del tumor
    
    **Para desbloquear la funcionalidad completa de IA:** Haz clic en "Descargar Modelos IA" en la barra lateral.
    
    **Mostrando actualmente:** Interfaz de demostración con imágenes de muestra
    """)
    model = None

# Initialize tumor segmentor with YOLO and SAM models (only if models are available)
segmentor = None
if models_available or st.session_state.models_downloaded:
    # Look for YOLO model
    yolo_model_paths = [
        os.path.join("models", "yolo_best.pt"),
        os.path.join("models", "best.pt")
    ]

    yolo_model_path = None
    for path in yolo_model_paths:
        if os.path.exists(path):
            yolo_model_path = path
            break

    # Look for SAM model  
    sam_model_paths = [
        os.path.join("models", "sam2_b.pt"),
        os.path.join("models", "sam2_1_hiera_large.pt"),
        os.path.join("models", "sam2_hiera_large.pt"),
        os.path.join("models", "sam2_1_hiera_l.pt"),
        os.path.join("models", "sam2_hiera_l.pt")
    ]

    sam_model_path = None
    for path in sam_model_paths:
        if os.path.exists(path):
            sam_model_path = path
            break

    try:
        # First make sure PyTorch is properly set up for model loading
        from src.pytorch_fix import allow_model_loading
        allow_model_loading()
        
        # Now try to create the tumor segmentor
        segmentor = TumorSegmentor(yolo_model_path=yolo_model_path, sam_model_path=sam_model_path, device=device)
        
        # Log the status of the segmentor
        if segmentor and segmentor.yolo_model and segmentor.sam_model:
            print("✅ Both YOLO and SAM models loaded successfully")
        elif segmentor and segmentor.yolo_model:
            print("⚠️ Only YOLO model loaded successfully, SAM failed")
        elif segmentor and segmentor.sam_model:
            print("⚠️ Only SAM model loaded successfully, YOLO failed")
        else:
            print("❌ Neither YOLO nor SAM models loaded successfully")
            
    except Exception as e:
        import traceback
        st.warning(f"Modelos de segmentación no disponibles: {str(e)}")
        print(f"Error initializing segmentation models: {e}")
        print(f"Error details: {traceback.format_exc()}")
        segmentor = None

# Define the transformation with error handling
try:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("✅ Transform pipeline created successfully")
except Exception as e:
    st.error(f"Falló la creación del pipeline de transformación: {e}")
    st.stop()

# map labels from int to string
label_dict = {
    0: "Sin Tumor",
    1: "Pituitario",
    2: "Glioma",
    3: "Meningioma",
    4: "Otro",
}

# process image got from user before passing to the model
def preprocess_image(image):
    try:
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms with error handling
        preprocessed_image = transform(image).unsqueeze(0)
        return preprocessed_image
    except Exception as e:
        print(f"Transform failed: {e}")
        st.warning(f"Usando preprocesamiento alternativo debido a: {e}")
        
        try:
            # Fallback: manual preprocessing
            import numpy as np
            
            # Manual resize
            image = image.resize((224, 224))
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Normalize manually
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std
            
            # Convert to tensor
            import torch
            tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
            return tensor
        except Exception as fallback_error:
            st.error(f"Ambos métodos de preprocesamiento fallaron: {fallback_error}")
            raise fallback_error

# sample image loader
@st.cache_data
def load_sample_images(sample_images_dir):
    sample_image_files = os.listdir(sample_images_dir)
    sample_images = []
    for sample_image_file in sample_image_files:
        sample_image_path = os.path.join(sample_images_dir, sample_image_file)
        sample_image = Image.open(sample_image_path).convert("RGB")
        sample_image = sample_image.resize((150, 150))  # Resize to a fixed size
        sample_images.append((sample_image_file, sample_image))
    return sample_images

# Streamlit app
st.title("🧠 Clasificación y Segmentación de Tumores Cerebrales")
st.markdown("---")

# Add information about YOLO+SAM2 models
col1, col2 = st.columns([2, 1])
with col1:
    if segmentor is not None:
        if segmentor.yolo_model is not None and segmentor.sam_model is not None:
            st.success("✅ Modelos YOLO+SAM2: Cargados y listos para segmentación avanzada")
        elif segmentor.yolo_model is not None or segmentor.sam_model is not None:
            st.warning("⚠️ Carga Parcial de Modelos: Algunos modelos cargados, usando enfoque híbrido")
        else:
            st.info("🔧 Modo Alternativo: Usando procesamiento mejorado de imágenes para segmentación")
            st.caption("Se están utilizando técnicas tradicionales de visión por computadora para la segmentación de tumores")
    else:
        if not (models_available or st.session_state.models_downloaded):
            st.info("🚀 Listo para descargar modelos IA para funcionalidad completa")
        else:
            st.warning("⚠️ Modelos de segmentación no disponibles")

with col2:
    st.info(f"🖥️ Dispositivo: {device.upper()}")

st.markdown("---")


# Display sample images section
st.subheader("Imágenes de Muestra")
st.write(
    "Aquí hay algunas imágenes de muestra. Tu imagen cargada debe ser similar a estas para obtener mejores resultados."
)

sample_images_dir = "sample"
sample_images = load_sample_images(sample_images_dir)

# Create a grid layout for sample images
num_cols = 3  # Number of columns in the grid
cols = st.columns(num_cols)

for i, (sample_image_file, sample_image) in enumerate(sample_images):
    col_idx = i % num_cols
    with cols[col_idx]:
        st.image(sample_image, caption=f"Muestra {i+1}", use_container_width=True)


st.write("Carga una imagen a continuación para clasificarla.")


# image from user
uploaded_file = st.file_uploader("Elige una imagen...", type="jpg")

if uploaded_file is not None:
    if not (models_available or st.session_state.models_downloaded):
        st.warning("⚠️ Modelos no disponibles. Solo modo demostración - mostrando interfaz sin predicción real.")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen Cargada", width=210)
        
        st.write("🔧 **Modo Demostración**: Así es como funcionaría la interfaz con los modelos cargados.")
        st.info("💡 ¡Haz clic en 'Descargar Modelos IA' en la barra lateral para desbloquear la funcionalidad completa!")
        
        # Show what the output would look like
        st.markdown("---")
        st.subheader("Salida Esperada (Demo)")
        st.write("🔍 **Clasificación**: Glioma, Meningioma, Pituitario, o Sin Tumor")
        st.write("📍 **Detección**: Cuadro delimitador alrededor del área del tumor")  
        st.write("🎯 **Segmentación**: Resaltado preciso del límite del tumor")
        
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen Cargada", width=210)

            # Preprocess the image
            print("Starting image preprocessing...")
            preprocessed_image = preprocess_image(image).to(device)
            print("Image preprocessing completed")
            
            # Make prediction
            print("Starting prediction...")
            predicted_class = predict(model, preprocessed_image, device)
            predicted_label = label_dict[predicted_class]
            print(f"Prediction completed: {predicted_label}")
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
            print(f"Error in main processing: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            st.stop()

        st.write(
            f"<h1 style='font-size: 48px;'>Predicción: {predicted_label}</h1>",
            unsafe_allow_html=True,
        )
        
        # Add segmentation visualization (only if segmentor is available)
        if segmentor is not None:
            st.subheader("Análisis de Segmentación del Tumor")
            
            with st.spinner("Generando visualización de segmentación..."):
                # Let's implement a more robust approach with proper fallback
                try:
                    print("Starting segmentation processing...")
                    
                    # First, let's check if it's a "No Tumor" prediction
                    if predicted_label == "Sin Tumor":
                        print("Processing No Tumor case...")
                        # For "No Tumor" cases, just use the basic visualization
                        segmentation_result = segmentor.process_image(image, predicted_label)
                        st.image(segmentation_result, caption="Resultado de Segmentación", use_container_width=True)
                        st.success("✅ No se detectó tumor - la imagen parece normal")
                    else:
                        print(f"Processing {predicted_label} case...")
                        # For tumor cases, try the advanced segmentation first
                        st.info("Intentando segmentación de tumor basada en IA...")
                        
                        try:
                            # Check if models are available
                            if segmentor.yolo_model is None or segmentor.sam_model is None:
                                print("Models not available, using fallback...")
                                st.warning("⚠️ Modelos avanzados de segmentación IA no están completamente disponibles")
                                st.info("Usando método de segmentación alternativo")
                                
                                # Use fallback method
                                fallback_result = create_fallback_visualization(image, predicted_label)
                                st.image(fallback_result, caption="Resultado de Segmentación Alternativa", use_container_width=True)
                                
                            else:
                                print("Attempting advanced segmentation...")
                                # Try advanced segmentation
                                segmentation_result = segmentor.process_image(image, predicted_label)
                                print("Advanced segmentation completed")
                                
                                # Look for failure indicators in the result
                                if isinstance(segmentation_result, str) and "failed" in segmentation_result.lower():
                                    print("Advanced segmentation returned failure, using fallback...")
                                    st.warning("⚠️ La segmentación IA avanzada falló")
                                    st.info("Usando método de segmentación alternativo")
                                    
                                    # Use fallback method
                                    fallback_result = create_fallback_visualization(image, predicted_label)
                                    st.image(fallback_result, caption="Resultado de Segmentación Alternativa", use_container_width=True)
                                    
                                else:
                                    # Show the advanced segmentation result
                                    st.image(segmentation_result, caption="Resultado de Segmentación IA", use_container_width=True)
                                    
                        except Exception as seg_error:
                            print(f"Error in advanced segmentation: {seg_error}")
                            st.warning(f"⚠️ Error de segmentación: {seg_error}")
                            st.info("Usando método de segmentación alternativo")
                            
                            # Use fallback method
                            try:
                                fallback_result = create_fallback_visualization(image, predicted_label)
                                st.image(fallback_result, caption="Resultado de Segmentación Alternativa", use_container_width=True)
                            except Exception as fallback_error:
                                st.error(f"Ambos métodos de segmentación fallaron: {fallback_error}")
                                print(f"Fallback segmentation also failed: {fallback_error}")
                                
                        # Common success message for all tumor cases (only if no errors)
                        st.success("🔍 Región del tumor resaltada con contorno rojo y relleno semi-transparente")
                        
                except Exception as e:
                    print(f"Critical error in segmentation section: {e}")
                    st.error(f"La visualización de segmentación falló: {e}")
                    st.info("El resultado de clasificación sigue siendo válido arriba.")
                    import traceback
                    print(f"Segmentation error traceback: {traceback.format_exc()}")
            
        else:
            print("Segmentor not available, skipping segmentation")
            st.info("💡 Modelos de segmentación no cargados. Resultado de clasificación mostrado arriba.")