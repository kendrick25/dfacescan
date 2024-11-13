import os

#lista de nombres de carpetas

folder_names = [
"aparicio-azner",
"ayala-javier",
"bernal-jaime",
"caballero-luis",
"delatorre-pedro",
"diaz-julian",
"freitas-jeremy",
"gonzalez-edwar",
"justavino-michelle",
"montenegro-orlando",
"paredes-jeisson",
"pimentel-sebastian",
"pinto-anibal",
"pitty-ketzy",
"reigosa-camilo",
"reyes-jean",
"ruiz-rene",
"saavedra-allan",
"samudio-thais",
"sanchez-kendrick",
"sanjur-ricardo",
"serrano-christopher",
"sobenis-dilan",
]

#directorio base (opcional, donde se crearán las carpetas)
base_directory = "../data/classroom/ref/"

#crear cada carpeta en la lista
for folder in folder_names:
    path = os.path.join(base_directory, folder)
    os.makedirs(path, exist_ok=True)  # crea la carpeta si no existe
    print(f"Directory '{folder}' created in '{path}'")

