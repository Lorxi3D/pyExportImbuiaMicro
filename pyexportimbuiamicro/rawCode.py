#!/usr/bin/env python
# coding: utf-8

# In[15]:


from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.io import DatMetaReader
from Orange.data import Table

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import olefile
import os
from io import BytesIO

HOME = os.path.expanduser("~") 


# In[16]:


def extract_optical_image_from_bsp(bsp_file_path):
    """
    Extrai a imagem óptica de um arquivo .bsp.

    Args:
    bsp_file_path (str): Caminho para o arquivo .bsp.

    Returns:
    np.array: Imagem óptica como um array NumPy.
    """
    jpg_byte_start = b'\xff\xd8\xff\xe0'
    jpg_byte_end = b'\xff\xd9'
    jpg_image = bytearray()

    with olefile.OleFileIO(bsp_file_path) as ole:
        streams = ole.listdir(storages=False, streams=True)
        for stream_name in streams:
            stream_data = ole.openstream(stream_name).read()
            startimg = stream_data.find(jpg_byte_start)
            if startimg == -1:
                continue
            else:
                endimg = stream_data.find(jpg_byte_end, startimg) + len(jpg_byte_end)
                jpg_image += stream_data[startimg:endimg]
                break

    if not jpg_image:
        raise ValueError("Nenhuma imagem JPEG encontrada no arquivo .bsp.")

    image = Image.open(BytesIO(jpg_image))
    return np.array(image)


# In[17]:


def load_and_save_data(file_path, hdf5_path, bsp_file_path, pixel_size=None):
    """
    Carrega dados usando o DatMetaReader e salva diretamente em um arquivo HDF5, incluindo a resolução espacial e a imagem óptica.

    Args:
    file_path (str): Caminho para o arquivo .dat.
    hdf5_path (str): Caminho para o arquivo HDF5 de destino.
    bsp_file_path (str): Caminho para o arquivo .bsp contendo a imagem óptica.
    pixel_size (tuple or None): Tamanho do pixel para a resolução espacial (x, y). Se None, a resolução será calculada a partir das coordenadas.
    """
    try:
        reader = DatMetaReader(file_path)
        data = reader.read()
        print("Dados carregados com sucesso!")

        if isinstance(data, Table):
            data_array = data.X  # .X acessa os dados em formato NumPy
            wavenumbers = getx(data)  # Supondo que getx(data) retorne os números de onda como array NumPy

            # Extraindo coordenadas (map_x, map_y) dos metadados
            map_x = data.get_column('map_x').astype(float)
            map_y = data.get_column('map_y').astype(float)
            coordinates = np.column_stack((map_x, map_y))

            # Definindo unique_x e unique_y para cálculo posterior
            unique_x = np.unique(map_x)
            unique_y = np.unique(map_y)

            # Calculando ou ajustando a resolução espacial
            if pixel_size is None:
                pixel_size_x = np.diff(unique_x).mean() if len(unique_x) > 1 else 0
                pixel_size_y = np.diff(unique_y).mean() if len(unique_y) > 1 else 0
                pixel_size = (pixel_size_x, pixel_size_y)
            else:
                # Ajustando as coordenadas com base na nova resolução
                coordinates[:, 0] *= pixel_size[0] / (np.diff(unique_x).mean() if len(unique_x) > 1 else 1)
                coordinates[:, 1] *= pixel_size[1] / (np.diff(unique_y).mean() if len(unique_y) > 1 else 1)

            # Calculando a resolução espectral
            spectral_resolution = np.diff(wavenumbers).mean() if len(wavenumbers) > 1 else 0

            # Calculando o tamanho da imagem em pixels e micrômetros
            image_size_pixels = (len(unique_x), len(unique_y))
            image_size_micrometers = (image_size_pixels[0] * pixel_size[0], image_size_pixels[1] * pixel_size[1])

            print(f"Spectra shape: {data_array.shape}")
            print(f"Wavenumbers shape: {wavenumbers.shape}")
            print(f"Coordinates shape: {coordinates.shape}")
            print(f"Pixel size: {pixel_size} µm")
            print(f"Spectral resolution: {spectral_resolution}")
            print(f"Image size (pixels): {image_size_pixels}")
            print(f"Image size (micrometers): {image_size_micrometers}")

        with h5py.File(hdf5_path, 'w') as h5f:
            h5f.create_dataset('spectra', data=data_array)
            h5f.create_dataset('wavenumbers', data=wavenumbers)
            h5f.create_dataset('coordinates', data=coordinates)
            h5f.attrs['pixel_size_x'] = pixel_size[0]
            h5f.attrs['pixel_size_y'] = pixel_size[1]
            h5f.attrs['spectral_resolution'] = spectral_resolution
            h5f.attrs['image_size_pixels'] = image_size_pixels
            h5f.attrs['image_size_micrometers'] = image_size_micrometers

            # Extraindo a imagem óptica do .bsp
            try:
                optical_image = extract_optical_image_from_bsp(bsp_file_path)
                h5f.create_dataset('optical_image', data=optical_image)
                print("Imagem óptica salva no HDF5 com sucesso!")
            except Exception as e:
                print(f"Erro ao extrair imagem óptica do .bsp: {e}")

        print("Dados salvos no HDF5 com sucesso!")

        return reader

    except Exception as e:
        print(f"Erro ao carregar ou salvar dados: {e}")


# In[18]:


def load_spectral_data(file_name):
    """
    Carrega dados espectrais, números de onda e coordenadas de um arquivo HDF5, incluindo a resolução espacial.

    Args:
        file_name (str): Caminho para o arquivo HDF5 de onde os dados serão carregados.

    Returns:
        tuple: Um tuple contendo:
            - data (np.array): Array de dados espectrais.
            - wavenumbers (np.array): Array de números de onda.
            - coordinates (np.array): Array de coordenadas (x, y).
            - pixel_size (tuple): Tamanho do pixel para a resolução espacial (x, y).
            - spectral_resolution (float): Resolução espectral.
            - image_size_pixels (tuple): Tamanho da imagem em pixels (x, y).
            - image_size_micrometers (tuple): Tamanho da imagem em micrometros (x, y).
            - optical_image (np.array or None): Imagem óptica, se disponível.
    """
    with h5py.File(file_name, 'r') as h5f:
        if 'spectra' in h5f:
            data = h5f['spectra'][:]
        else:
            raise KeyError("Dataset 'spectra' não encontrado no arquivo HDF5.")

        if 'wavenumbers' in h5f:
            wavenumbers = h5f['wavenumbers'][:]
        else:
            raise KeyError("Números de onda não encontrados no arquivo HDF5.")

        if 'coordinates' in h5f:
            coordinates = h5f['coordinates'][:]
        else:
            raise KeyError("Coordenadas não encontradas no arquivo HDF5.")
        
        pixel_size_x = h5f.attrs.get('pixel_size_x', None)
        pixel_size_y = h5f.attrs.get('pixel_size_y', None)
        pixel_size = (pixel_size_x, pixel_size_y)

        spectral_resolution = h5f.attrs.get('spectral_resolution', None)
        image_size_pixels = h5f.attrs.get('image_size_pixels', None)
        image_size_micrometers = h5f.attrs.get('image_size_micrometers', None)

        optical_image = h5f['optical_image'][:] if 'optical_image' in h5f else None

    return data, wavenumbers, coordinates, pixel_size, spectral_resolution, image_size_pixels, image_size_micrometers, optical_image


# In[19]:


def visualize_optical_image(optical_image):
    """
    Visualiza a imagem óptica.

    Args:
    optical_image (np.array): Imagem óptica como um array NumPy.
    """
    plt.imshow(optical_image)
    plt.axis('off')
    plt.title('Imagem Óptica')
    plt.show()


# In[20]:


def save_optical_image(optical_image, save_path):
    """
    Salva a imagem óptica em um arquivo PNG.

    Args:
    optical_image (np.array): Imagem óptica como um array NumPy.
    save_path (str): Caminho para salvar a imagem.
    """
    img = Image.fromarray(optical_image)
    img.save(save_path)
    print(f"Imagem óptica salva em {save_path}")

# Caminho para os arquivos .dat, .h5, e .bsp
file_path = '/home/john/Documentos/Doutorado/pesq/minerais/mineraDados/infraTents/extracao/cto_25x/cto_25x.dat'
bsp_file_path = '/home/john/Documentos/Doutorado/pesq/minerais/mineraDados/infraTents/extracao/cto_25x/cto_25x.bsp'
pixel_size = (3.3, 3.3)  # Definindo a resolução manualmente, se necessário

hdf5_path = '/home/john/Documentos/Doutorado/pesq/minerais/mineraDados/infraTents/extracao/ext/spectra_data.h5'

original = load_and_save_data(file_path, hdf5_path, bsp_file_path, pixel_size)
# In[41]:


file_path = HOME + "/LorxiCloud/ProCatalis/20240520_sirius_imbuia(original)/BO_SUZANO/BO_SUZANO.dat"
bsp_file_path = HOME + "/LorxiCloud/ProCatalis/20240520_sirius_imbuia(original)/BO_SUZANO/BO_SUZANO.bsp"
hdf5_path = HOME + "/LorxiCloud/ProCatalis/20240520_sirius_imbuia_extracao/BioOil/datas/BO_SUZANO.h5"

original = load_and_save_data(file_path, hdf5_path, bsp_file_path)


# In[25]:


# Carregar os dados do HDF5
data, wavenumbers, coordinates, pixel_size, spectral_resolution, image_size_pixels, image_size_micrometers, optical_image = load_spectral_data(hdf5_path)
print(f"Pixel size: {pixel_size} µm")
print(f"Spectral resolution: {spectral_resolution}")
print(f"Image size (pixels): {image_size_pixels}")
print(f"Image size (micrometers): {image_size_micrometers}")
if optical_image is not None:
    print(f"Optical image shape: {optical_image.shape}")
    visualize_optical_image(optical_image)
    save_optical_image(optical_image, './images/BO_SUZANO.png')
else:
    print("No optical image found in the HDF5 file.")


# In[26]:


wavenumbers


# In[27]:


coordinates[2]


# In[34]:


plt.plot(wavenumbers, data[200])


# # Others Metadatas

# In[ ]:


import olefile
import re

def list_streams(bsp_file_path):
    """
    Lista todos os streams presentes no arquivo .bsp.

    Args:
    bsp_file_path (str): Caminho para o arquivo .bsp.
    """
    with olefile.OleFileIO(bsp_file_path) as ole:
        streams = ole.listdir()
        return streams

def extract_metadata_from_bsp(bsp_file_path):
    """
    Extrai metadados específicos de um arquivo .bsp.

    Args:
    bsp_file_path (str): Caminho para o arquivo .bsp.

    Returns:
    dict: Dicionário contendo os metadados extraídos.
    """
    metadata = {}
    keywords = ['Date', 'Time', 'Instrument', 'Mode', 'Measure', 'Resolution']
    pattern = re.compile(r'\b(?:' + '|'.join(keywords) + r')\b', re.IGNORECASE)

    with olefile.OleFileIO(bsp_file_path) as ole:
        streams = ole.listdir()
        for stream in streams:
            if isinstance(stream, tuple):
                stream = '/'.join(stream)
            if ole.exists(stream):
                with ole.openstream(stream) as stream_file:
                    data = stream_file.read().decode('utf-8', errors='ignore')
                    for line in data.split('\n'):
                        if pattern.search(line):
                            key_value = line.split(':', 1)
                            if len(key_value) == 2:
                                key, value = key_value
                                key = key.strip()
                                value = value.strip()
                                if key in metadata:
                                    metadata[key].append(value)
                                else:
                                    metadata[key] = [value]

    return metadata


# In[32]:


# Caminho para o arquivo .bsp
bsp_file_path = bsp_file_path

# Listar todos os streams
streams = list_streams(bsp_file_path)
print("Streams encontrados no arquivo .bsp:")
for stream in streams:
    print(stream)

# Extrair e exibir metadados específicos
metadata = extract_metadata_from_bsp(bsp_file_path)
print("\nMetadados extraídos:")
for key, values in metadata.items():
    print(f"{key}:")
    for value in values:
        print(f"  {value}")


# In[35]:


import olefile
import re

def extract_metadata_from_bsp(bsp_file_path, keywords):
    """
    Extrai metadados específicos de um arquivo .bsp.

    Args:
    bsp_file_path (str): Caminho para o arquivo .bsp.

    Returns:
    dict: Dicionário contendo os metadados extraídos, com cada palavra-chave mapeando para uma lista de strings encontradas ou uma string única.
    """
    # Inicializa o dicionário com cada palavra-chave mapeando para uma lista vazia
    metadata = {keyword: [] for keyword in keywords}

    with olefile.OleFileIO(bsp_file_path) as ole:
        streams = ole.listdir()
        for stream in streams:
            if isinstance(stream, tuple):
                stream = '/'.join(stream)
            if ole.exists(stream):
                with ole.openstream(stream) as stream_file:
                    data = stream_file.read().decode('utf-8', errors='ignore')
                    lines = data.split('\n')
                    for line in lines:
                        for keyword in keywords:
                            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                            match = pattern.search(line)
                            if match:
                                # Extrair o valor após o termo encontrado e limpá-lo
                                key_value = line.split(':', 1)
                                if len(key_value) == 2:
                                    value = key_value[1].strip()
                                else:
                                    value = line[match.end():].strip()

                                # Adicionar o valor encontrado à lista da palavra-chave correspondente
                                metadata[keyword].append(value)

    # Converter listas com um único item em uma única string
    for key in metadata:
        if len(metadata[key]) == 1:
            metadata[key] = metadata[key][0]

    return metadata


# In[36]:


def clean_metadata(metadata):
    """
    Limpa e formata o dicionário de metadados.

    Args:
    metadata (dict): Dicionário de metadados.

    Returns:
    dict: Dicionário de metadados limpo e formatado.
    """
    cleaned_metadata = {}
    for key, values in metadata.items():
        cleaned_values = []
        for value in values:
            # Remove caracteres não legíveis
            cleaned_value = re.sub(r'[^\x20-\x7E]', ' ', value)
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
            if cleaned_value and cleaned_value not in cleaned_values:
                cleaned_values.append(cleaned_value)
        cleaned_metadata[key] = cleaned_values
    return cleaned_metadata


# In[37]:


# Caminho para o arquivo .bsp
bsp_file_path = bsp_file_path

# Extrair e exibir metadados específicos
metadata = extract_metadata_from_bsp(bsp_file_path, ['Time', 'Mode', 'Resolution', 'ID', 'Serial Number', 'File Name', 'Objective'])
cleaned_metadata = clean_metadata(metadata)
print("\nMetadados extraídos:")
for key, values in cleaned_metadata.items():
    print(f"{key}:")
    for value in values:
        print(f"  {value}")


# In[38]:


['Time', 'Mode', 'Resolution', 'ID', 'Serial Number', 'File Name', 'Objective']


# In[39]:


# Caminho para o arquivo .bsp
bsp_file_path = bsp_file_path

# Extrair e exibir metadados específicos
metadata = extract_metadata_from_bsp(bsp_file_path, ['Time', 'Mode', 'Resolution', 'ID', 'Serial Number', 'File Name', 'Objective'])


# In[40]:


metadata['Time']


# In[ ]:




