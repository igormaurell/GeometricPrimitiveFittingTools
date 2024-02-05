#!/bin/sh
# Lista fixa de nomes de pastas
primeiro_argumento="$1"
pastas=("444_7k" "444_14k" "444_28k")
pastas2=("data" "eff_ransac_result_comp" "eff_ransac_result_max" "parsenet_pred_max" "primitivenet_pred_max")

lista_arquivos=()
for folder_name in "${pastas[@]}"; do
    for folder_name2 in "${pastas2[@]}"; do
        # Criar o caminho do diretório usando o template
        directory_path="/home/aeroscan/hd/LS3DC/dataset_pccs_merged_${folder_name}_after_noise/ls3dc/eval/${folder_name2}/seg/${primeiro_argumento}_instances.obj"
        lista_arquivos+=("$directory_path")
    done
done

for folder_name in "${pastas[@]}"; do
    for folder_name2 in "${pastas2[@]}"; do
        # Criar o caminho do diretório usando o template
        directory_path="/home/aeroscan/hd/LS3DC/dataset_pccs_merged_${folder_name}_after_noise/ls3dc/eval/${folder_name2}/seg/${primeiro_argumento}_types.obj"
        lista_arquivos+=("$directory_path")
    done
done

python o3d_view.py "${lista_arquivos[@]}"