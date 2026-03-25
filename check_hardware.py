import torch

def check_hardware():
    print("--- Comprobación de Hardware para Green AI ---")
    devices_available = []
    # Comprobar NVIDIA (CUDA)
    if torch.cuda.is_available():
        print(f"NVIDIA GPU detectada: {torch.cuda.get_device_name(0)}")
        devices_available.append("cuda")

    # Comprobar Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) detectada.")
        devices_available.append("mps")

    # Caer en CPU
    print("CPU detectada.")
    devices_available.append("cpu")

    print(f"\nDispositivos disponibles: '{devices_available}'")

    return devices_available

if __name__ == "__main__":
    check_hardware()