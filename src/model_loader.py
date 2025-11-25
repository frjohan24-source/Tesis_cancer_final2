"""
Módulo personalizado para cargar modelos YOLO y SAM con correcciones de compatibilidad
para PyTorch 2.6+ y modelos SAM2 en Streamlit Cloud.
"""
import os
import shutil
import torch
import sys
import warnings
import importlib.util

# Suprimir advertencias que podrían saturar la salida
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def is_module_available(module_name):
    """Verificar si un módulo está disponible/instalado"""
    return importlib.util.find_spec(module_name) is not None

def load_yolo_model(yolo_model_path):
    """Cargar modelo YOLO con correcciones de compatibilidad para PyTorch 2.6+"""
    # Primero verificar si ultralytics está disponible
    if not is_module_available("ultralytics"):
        print("❌ Módulo ultralytics no encontrado. Instalando...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.145"])
            print("✅ Ultralytics instalado exitosamente")
        except Exception as e:
            print(f"❌ Falló la instalación de ultralytics: {e}")
            return None

    try:
        # Importar YOLO después de asegurar que ultralytics está instalado
        from ultralytics import YOLO
        
        # Verificación de versión de PyTorch
        pytorch_version = torch.__version__.split('.')
        major, minor = int(pytorch_version[0]), int(pytorch_version[1])
        is_pt26_plus = (major > 2) or (major == 2 and minor >= 6)
        
        if is_pt26_plus:
            print(f"Detectado PyTorch {torch.__version__} (tiene seguridad weights_only)")
            # Asegurar que nuestro cargador parcheado está activo
            try:
                from src.pytorch_fix import allow_model_loading
                allow_model_loading()
            except ImportError:
                # Intentar ruta de importación diferente
                try:
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from pytorch_fix import allow_model_loading
                    allow_model_loading()
                except Exception as e:
                    print(f"⚠️ No se pudo aplicar corrección de PyTorch: {e}")
        
        # Enfoque de carga multi-estrategia
        strategies = [
            # Estrategia 1: Carga directa
            lambda: YOLO(yolo_model_path),
            
            # Estrategia 2: Carga manual con weights_only=False
            lambda: YOLO(torch.load(yolo_model_path, weights_only=False, map_location='cpu')),
            
            # Estrategia 3: Intentar un método diferente del constructor YOLO
            lambda: YOLO(model=yolo_model_path),
            
            # Estrategia 4: Archivo temporal con extensión .yaml
            lambda: try_yaml_extension(yolo_model_path, YOLO),
            
            # Estrategia 5: Descargar modelo preentrenado
            lambda: YOLO('yolov8n.pt')
        ]
        
        # Intentar cada estrategia en orden
        for i, strategy in enumerate(strategies):
            try:
                print(f"Intentando estrategia de carga YOLO {i+1}...")
                model = strategy()
                print(f"✅ Modelo YOLO cargado exitosamente con estrategia {i+1}!")
                return model
            except Exception as e:
                print(f"⚠️ Estrategia {i+1} falló: {str(e)[:100]}...")
        
        print("❌ Todas las estrategias de carga YOLO fallaron")
        return None
    
    except Exception as e:
        print(f"❌ Error fatal al cargar YOLO: {e}")
        return None

def try_yaml_extension(model_path, YOLO_class):
    """Intentar cargar creando un archivo temporal con extensión .yaml"""
    try:
        # Crear un archivo temporal con extensión .yaml
        yaml_path = model_path + '.yaml'
        shutil.copy2(model_path, yaml_path)
        model = YOLO_class(yaml_path)
        # Limpiar
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
        return model
    except Exception as e:
        print(f"⚠️ Enfoque YAML falló: {e}")
        # Limpiar cualquier archivo temporal
        if os.path.exists(model_path + '.yaml'):
            os.remove(model_path + '.yaml')
        raise e

def load_sam_model(sam_model_path):
    """Cargar modelo SAM con correcciones de compatibilidad para modelos SAM2"""
    # Primero verificar si ultralytics está disponible
    if not is_module_available("ultralytics"):
        print("❌ Módulo ultralytics no encontrado. Instalando...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.145"])
            print("✅ Ultralytics instalado exitosamente")
        except Exception as e:
            print(f"❌ Falló la instalación de ultralytics: {e}")
            return None
            
    try:
        from ultralytics import SAM
        
        # Enfoque multi-estrategia para SAM
        strategies = [
            # Estrategia 1: Carga directa
            lambda: try_direct_sam_loading(sam_model_path, SAM),
            
            # Estrategia 2: Renombrar a nombre estándar y cargar
            lambda: try_renamed_sam_loading(sam_model_path, SAM),
            
            # Estrategia 3: Descargar modelo oficial
            lambda: SAM('sam_b.pt')
        ]
        
        # Intentar cada estrategia en orden
        for i, strategy in enumerate(strategies):
            try:
                print(f"Intentando estrategia de carga SAM {i+1}...")
                model = strategy()
                if model:
                    print(f"✅ Modelo SAM cargado exitosamente con estrategia {i+1}!")
                    return model
            except Exception as e:
                print(f"⚠️ Estrategia SAM {i+1} falló: {str(e)[:100]}...")
        
        print("❌ Todas las estrategias de carga SAM fallaron")
        return None
            
    except Exception as e:
        print(f"❌ Error fatal al cargar SAM: {e}")
        return None

def try_direct_sam_loading(sam_model_path, SAM_class):
    """Intentar carga directa del modelo SAM"""
    try:
        return SAM_class(sam_model_path)
    except Exception:
        return None

def try_renamed_sam_loading(sam_model_path, SAM_class):
    """Intentar cargar modelo SAM renombrándolo a formato estándar"""
    try:
        # Crear una copia renombrada con nombre estándar
        standard_sam_paths = [
            os.path.join(os.path.dirname(sam_model_path), "sam_b.pt"),
            os.path.join(os.path.dirname(sam_model_path), "sam_l.pt"),
            os.path.join(os.path.dirname(sam_model_path), "mobile_sam.pt")
        ]
        
        # Intentar cada nombre estándar
        for standard_path in standard_sam_paths:
            try:
                # Solo copiar si el destino no existe
                if not os.path.exists(standard_path):
                    shutil.copy2(sam_model_path, standard_path)
                    print(f"Copia creada en {standard_path}")
                
                # Intentar cargar el modelo renombrado
                model = SAM_class(standard_path)
                print(f"✅ Modelo SAM cargado desde copia renombrada: {standard_path}")
                return model
            except Exception:
                continue
        
        return None
    except Exception:
        return None
