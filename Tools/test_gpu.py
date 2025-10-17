import os
import tensorflow as tf

# Adiciona o caminho da DLL do CUDA explicitamente antes de importar o TensorFlow
# Ajuste o caminho se a sua instalação for diferente
cuda_bin_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin"
if os.path.exists(cuda_bin_path):
    print(f"Adicionando caminho da DLL: {cuda_bin_path}")
    os.add_dll_directory(cuda_bin_path)
else:
    print(f"AVISO: O caminho da DLL do CUDA não foi encontrado em {cuda_bin_path}")


print(f"Versão do TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ Sucesso! GPU(s) encontradas: {gpus}")
else:
    print("❌ Falha Definitiva: Nenhuma GPU detectada pelo TensorFlow após todas as verificações.")
