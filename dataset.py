import os
from glob import glob
from PIL import Image
import py7zr
import torchvision.transforms.v2 as tfs_v2
import torch.utils.data as data



class FabricDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        # Si recibido un archivo .7z, abriendolo en una carpeta temporal
        if image_dir.endswith('.7z'):
            self.extract_dir = image_dir.replace('.7z', '_extracted')
            if not os.path.exists(self.extract_dir):
                print(f"Desempaquetando el archivo {image_dir}...")
                with py7zr.SevenZipFile(image_dir, mode='r') as archive:
                    archive.extractall(path=self.extract_dir)
            image_dir = self.extract_dir

        # Guardar rutas de todas las imágenes PNG/JPG/JPEG/TIF de la carpeta
        self.image_paths = glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

        if not self.image_paths:
            raise RuntimeError(f'Изображения не найдены в {image_dir}')

        if transform is None:
            self.transform = tfs_v2.Compose([
                tfs_v2.Resize(size=(256, 256)),
                tfs_v2.ToTensor(),
                tfs_v2.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Obtener la ruta a la imagen por index.
        image_path = self.image_paths[idx]

        # Abrir la imagen mediante PIL y la convertir a RGB.
        image = Image.open(image_path).convert('RGB')

        # Aplicar transformaciones
        image = self.transform(image)

        return image

