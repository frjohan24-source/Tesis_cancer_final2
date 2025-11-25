import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import sys
import traceback
import shutil

# Intentar importar nuestro cargador de modelos personalizado
try:
    from src.model_loader import load_yolo_model, load_sam_model
    MODEL_LOADER_AVAILABLE = True
    print("✅ Cargador de modelos personalizado importado exitosamente")
except ImportError:
    print("⚠️ No se pudo importar el cargador de modelos personalizado, intentando ruta directa...")
    try:
        # Intentar con ruta de importación relativa diferente
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model_loader import load_yolo_model, load_sam_model
        MODEL_LOADER_AVAILABLE = True
        print("✅ Cargador de modelos personalizado importado vía ruta alternativa")
    except ImportError as e:
        print(f"❌ Falló la importación del cargador de modelos personalizado: {e}")
        MODEL_LOADER_AVAILABLE = False

# Intentar importar ultralytics
try:
    from ultralytics import YOLO, SAM
    ULTRALYTICS_AVAILABLE = True
    print("✅ Ultralytics YOLO y SAM cargados exitosamente")
except Exception as e:
    print(f"❌ Error de importación de Ultralytics: {str(e)}")
    print(f"Error detallado: {traceback.format_exc()}")
    ULTRALYTICS_AVAILABLE = False
    
    # Intentar instalar ultralytics si no está disponible
    try:
        import subprocess
        print("Instalando ultralytics vía pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.208"])
        from ultralytics import YOLO, SAM
        ULTRALYTICS_AVAILABLE = True
        print("✅ Ultralytics instalado y cargado exitosamente")
    except Exception as e2:
        print(f"❌ Falló la instalación de Ultralytics: {str(e2)}")
        ULTRALYTICS_AVAILABLE = False

class TumorSegmentor:
    def __init__(self, yolo_model_path=None, sam_model_path=None, device="cpu"):
        self.device = device
        self.yolo_model = None
        self.sam_model = None
        
        if ULTRALYTICS_AVAILABLE:
            self.load_models(yolo_model_path, sam_model_path)
        else:
            print("⚠️ Ultralytics no disponible. La segmentación usará el método alternativo.")
    
    def load_models(self, yolo_model_path, sam_model_path):
        """Cargar modelos de detección YOLO y segmentación SAM"""
        try:
            # Cargar modelo YOLO para detección de tumores
            if yolo_model_path and os.path.exists(yolo_model_path):
                print(f"🔧 Cargando modelo YOLO desde: {yolo_model_path}")
                
                # Usar cargador de modelos personalizado si está disponible
                if MODEL_LOADER_AVAILABLE:
                    print("Usando cargador de modelos personalizado para YOLO")
                    self.yolo_model = load_yolo_model(yolo_model_path)
                    if self.yolo_model:
                        print("✅ Modelo YOLO cargado exitosamente con cargador personalizado!")
                    else:
                        print("❌ Falló la carga personalizada del modelo YOLO")
                else:
                    # Alternativa a carga directa si el cargador personalizado no está disponible
                    try:
                        # Primero aplicar corrección de PyTorch 2.6+ para weights_only
                        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                            try:
                                # Importar las clases necesarias
                                from ultralytics.nn.tasks import DetectionModel
                                # Agregarlas a la lista de globales seguros
                                torch.serialization.add_safe_globals([DetectionModel])
                                print("✅ DetectionModel agregado a globales seguros de PyTorch")
                            except Exception as e:
                                print(f"⚠️ No se pudo agregar a globales seguros: {e}")
                                
                        # Intentar cargar con enfoque estándar primero
                        self.yolo_model = YOLO(yolo_model_path)
                        print("✅ Modelo YOLO cargado con enfoque estándar!")
                    except Exception as e:
                        print(f"❌ Falló la carga estándar de YOLO: {e}")
                        self.yolo_model = None
            else:
                print(f"⚠️ Modelo YOLO no encontrado en: {yolo_model_path}")
                self.yolo_model = None
            
            # Cargar modelo SAM para segmentación
            if sam_model_path and os.path.exists(sam_model_path):
                print(f"🔧 Cargando modelo SAM desde: {sam_model_path}")
                
                # Usar cargador de modelos personalizado si está disponible
                if MODEL_LOADER_AVAILABLE:
                    print("Usando cargador de modelos personalizado para SAM")
                    self.sam_model = load_sam_model(sam_model_path)
                    if self.sam_model:
                        print("✅ Modelo SAM cargado exitosamente con cargador personalizado!")
                    else:
                        print("❌ Falló la carga personalizada del modelo SAM")
                else:
                    # Manejar carga del modelo SAM2 con enfoque directo
                    if "sam2" in os.path.basename(sam_model_path).lower():
                        # Crear una copia renombrada con nombre estándar
                        standard_sam_path = os.path.join(os.path.dirname(sam_model_path), "sam_b.pt")
                        try:
                            # Solo copiar si el destino no existe
                            if not os.path.exists(standard_sam_path):
                                shutil.copy2(sam_model_path, standard_sam_path)
                                print(f"✅ Copia compatible creada en {standard_sam_path}")
                            
                            # Intentar cargar el modelo renombrado
                            self.sam_model = SAM(standard_sam_path)
                            print("✅ Modelo SAM cargado desde copia renombrada!")
                        except Exception as e:
                            print(f"❌ Falló el renombrado de SAM: {e}")
                            self.sam_model = None
                    else:
                        # Carga estándar del modelo SAM
                        try:
                            self.sam_model = SAM(sam_model_path)
                            print("✅ Modelo SAM cargado con enfoque estándar!")
                        except Exception as e:
                            print(f"❌ Falló la carga estándar de SAM: {e}")
                            self.sam_model = None
            else:
                print(f"⚠️ Modelo SAM no encontrado en: {sam_model_path}")
                self.sam_model = None
                
        except Exception as e:
            print(f"❌ Error en el proceso de carga de modelos: {e}")
            import traceback
            print(f"Error detallado de carga de modelos: {traceback.format_exc()}")
            print("🔄 Los modelos usarán el método de segmentación alternativo")
            self.yolo_model = None
            self.sam_model = None
    
    def detect_tumor_with_yolo(self, image):
        """Usar YOLO para detectar regiones de tumores y devolver caja delimitadora"""
        if self.yolo_model is None:
            return None
        
        try:
            # Convertir PIL a array numpy para YOLO
            img_array = np.array(image)
            
            # Ejecutar detección YOLO
            results = self.yolo_model(img_array, verbose=False)
            
            # Obtener el primer resultado
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Obtener la caja con mayor confianza
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                
                # Obtener coordenadas de la caja delimitadora (x1, y1, x2, y2)
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                confidence = confidences[best_idx]
                
                print(f"🎯 YOLO detectó tumor con confianza: {confidence:.3f}")
                return bbox, confidence
            else:
                print("🔍 YOLO no detectó ningún tumor")
                return None
                
        except Exception as e:
            print(f"❌ Error en detección YOLO: {e}")
            return None
    
    def segment_with_sam(self, image, bbox):
        """Usar SAM para segmentar el tumor basado en la detección de YOLO"""
        if self.sam_model is None:
            return None
        
        try:
            # Convertir PIL a array numpy
            img_array = np.array(image)
            
            # Convertir bbox al formato que SAM espera [x1, y1, x2, y2]
            # bbox ya está en el formato correcto desde YOLO
            bbox_tensor = torch.tensor(bbox).unsqueeze(0)  # Agregar dimensión de lote
            
            # Ejecutar segmentación SAM con prompt de caja delimitadora (como el SAM original funcional)
            results = self.sam_model(img_array, bboxes=bbox_tensor, verbose=False)
            
            if len(results) > 0 and results[0].masks is not None:
                # Obtener la máscara
                mask = results[0].masks.data[0].cpu().numpy()
                
                # Convertir a formato uint8
                mask = (mask * 255).astype(np.uint8)
                
                print("✅ Segmentación SAM exitosa")
                return mask
            else:
                print("⚠️ SAM no generó una máscara")
                return None
                
        except Exception as e:
            print(f"❌ Error en segmentación SAM: {e}")
            return None
    
    def segment_tumor(self, image, predicted_class):
        """
        Método principal de segmentación usando pipeline YOLO+SAM2
        """
        # Si no se predijo tumor, devolver None
        if predicted_class == "Sin Tumor":
            return None
        
        # Verificar si los modelos están disponibles
        if self.yolo_model is None:
            print("❌ Modelo YOLO no disponible para segmentación")
            return "FAILED: YOLO model not available"
            
        if self.sam_model is None:
            print("❌ Modelo SAM no disponible para segmentación")
            return "FAILED: SAM model not available"
        
        # Intentar pipeline YOLO+SAM 
        # Paso 1: Detectar con YOLO
        detection_result = self.detect_tumor_with_yolo(image)
        
        if detection_result is not None:
            bbox, confidence = detection_result
            
            # Paso 2: Segmentar con SAM
            mask = self.segment_with_sam(image, bbox)
            
            if mask is not None:
                return mask
            else:
                print("🔄 Segmentación SAM falló, no se generó máscara")
                return "FAILED: SAM segmentation generated no mask"
        else:
            print("🔄 Detección YOLO falló, no se detectaron regiones de tumor")
            return "FAILED: YOLO detection found no tumor regions"
    
        # Esta línea nunca debería alcanzarse pero se mantiene por seguridad
        return "FAILED: Unknown segmentation error"
    
    def create_segmented_visualization(self, original_image, mask, predicted_class, confidence=None):
        """
        Crear visualización con contorno rojo y relleno para la región del tumor
        """
        # Convertir PIL a array numpy
        img_array = np.array(original_image)
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Imagen original
        ax1.imshow(img_array)
        ax1.set_title("Imagen Original")
        ax1.axis('off')
        
        # Imagen segmentada
        ax2.imshow(img_array)
        
        # Manejar diferentes escenarios de máscara
        if isinstance(mask, str) and mask.startswith("FAILED:"):
            # Este es un mensaje de fallo, mostrarlo
            failure_reason = mask.split("FAILED:")[1].strip()
            ax2.text(0.5, 0.5, f"Segmentación no disponible\n({failure_reason})", 
                    transform=ax2.transAxes, fontsize=12, 
                    color='orange', ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        elif mask is not None and not isinstance(mask, str) and np.any(mask > 0):
            # Tenemos una máscara válida con contenido
            # Encontrar contornos para el contorno
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Crear superposición
            overlay = np.zeros_like(img_array)
            
            for contour in contours:
                # Rellenar el área del contorno con rojo semi-transparente
                cv2.fillPoly(overlay, [contour], (255, 0, 0))
                
                # Dibujar contorno rojo
                contour_points = contour.reshape(-1, 2)
                for i in range(len(contour_points)):
                    start_point = tuple(contour_points[i])
                    end_point = tuple(contour_points[(i + 1) % len(contour_points)])
                    ax2.plot([start_point[0], end_point[0]], 
                            [start_point[1], end_point[1]], 'r-', linewidth=2)
            
            # Aplicar superposición semi-transparente
            alpha = 0.3
            segmented_img = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
            ax2.imshow(segmented_img)
        else:
            # No hay segmentación disponible - mostrar imagen original con texto
            ax2.text(0.5, 0.5, "Segmentación no disponible\n(Detección YOLO+SAM2 falló)", 
                    transform=ax2.transAxes, fontsize=12, 
                    color='orange', ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        title = f"Resultado de Segmentación - {predicted_class}"
        if confidence:
            title += f" (Confianza: {confidence:.2f})"
        ax2.set_title(title)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Convertir gráfico a imagen
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()
        
        # Convertir a Imagen PIL
        result_image = Image.open(buf)
        return result_image
    
    def process_image(self, image, predicted_class, confidence=None):
        """
        Función principal de procesamiento
        """
        # Solo segmentar si se detectó tumor
        if predicted_class == "Sin Tumor":
            # Devolver imagen original con superposición de texto
            return self.create_no_tumor_visualization(image, predicted_class)
        
        # Realizar segmentación
        mask = self.segment_tumor(image, predicted_class)
        
        # Verificar si la segmentación falló (la máscara será una cadena que comienza con "FAILED:")
        if isinstance(mask, str) and mask.startswith("FAILED:"):
            print(f"Segmentación falló: {mask}")
            # Pasaremos el mensaje de fallo a la función de visualización
        
        # Crear visualización
        result_image = self.create_segmented_visualization(
            image, mask, predicted_class, confidence
        )
        
        return result_image
    
    def create_no_tumor_visualization(self, image, predicted_class):
        """
        Crear visualización para casos sin tumor
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(np.array(image))
        ax.set_title(f"Resultado: {predicted_class}")
        ax.axis('off')
        
        # Agregar superposición de texto
        ax.text(0.5, 0.05, "No se detectó tumor", 
                transform=ax.transAxes, fontsize=14, 
                color='green', ha='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Convertir a Imagen PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()
        
        result_image = Image.open(buf)
        return result_image