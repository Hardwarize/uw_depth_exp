import pkgutil
import importlib
from .data_provider import DataProvider

DataProviders = []

# Itera sobre todos los módulos (archivos .py) en el directorio actual
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Evita importar el archivo base o a sí mismo si fuera necesario, 
    # aunque la comprobación de instancia filtra bien.
    if module_name == "data_provider":
        continue
        
    # Importa el módulo dinámicamente
    module = importlib.import_module(f".{module_name}", package=__name__)
    
    # Inspecciona las variables del módulo
    for attributes_name, attribute_value in vars(module).items():
        # Si es una instancia de DataProvider, agrégala a la lista
        if isinstance(attribute_value, DataProvider):
            DataProviders.append(attribute_value)