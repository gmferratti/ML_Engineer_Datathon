#!/bin/bash
# Script para gerar um arquivo com o conteúdo de todos os arquivos úteis para o chatgpt

# Nome do arquivo de saída
output_file="arquivos_chatgpt.txt"

# Limpa o conteúdo do arquivo de saída (ou cria um novo)
> "$output_file"

# Gera a lista de arquivos úteis para o chatgpt (ignorando a pasta notebooks)
files=$(find . -type f \
  -not -path "./notebooks/*" \
  \( -name "LICENSE" -o -name "Makefile" -o -name "README.md" -o -name "TODO" -o -name "pyproject.toml" -o -name "*.py" \))

# Itera sobre cada arquivo e salva seu conteúdo no arquivo de saída
for file in $files; do
    echo "===== Conteúdo de $file =====" >> "$output_file"
    cat "$file" >> "$output_file"
    echo -e "\n\n" >> "$output_file"
done

echo "Conteúdo salvo em $output_file"
