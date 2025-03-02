#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}  Deploy do Sistema de Recomendação na AWS ${NC}"
echo -e "${BLUE}===========================================${NC}"

# Verificar dependências
echo -e "${YELLOW}Verificando dependências...${NC}"
command -v aws >/dev/null 2>&1 || { echo -e "${RED}AWS CLI não encontrado. Instale-o primeiro.${NC}"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker não encontrado. Instale-o primeiro.${NC}"; exit 1; }

# Verificar autenticação da AWS
echo -e "${YELLOW}Verificando autenticação AWS...${NC}"
aws sts get-caller-identity > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro na autenticação AWS. Execute 'aws configure' para configurar suas credenciais.${NC}"
    exit 1
fi

# Obter ID da conta AWS
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}AWS Account ID: ${AWS_ACCOUNT_ID}${NC}"

# Definir região (default: us-east-1)
read -p "Região AWS [us-east-1]: " AWS_REGION
AWS_REGION=${AWS_REGION:-us-east-1}

# Verificar se o repositório já existe
REPO_EXISTS=$(aws ecr describe-repositories --region $AWS_REGION --repository-names news-recommender-api 2>/dev/null)
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Criando repositório ECR...${NC}"
    aws ecr create-repository --repository-name news-recommender-api --region $AWS_REGION
else
    echo -e "${GREEN}Repositório ECR já existe.${NC}"
fi

# Autenticar no ECR
echo -e "${YELLOW}Autenticando no ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Construir imagem Docker
echo -e "${YELLOW}Construindo imagem Docker...${NC}"
docker build -t news-recommender-api .

# Tagear imagem para o ECR
echo -e "${YELLOW}Etiquetando imagem para o ECR...${NC}"
docker tag news-recommender-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/news-recommender-api:latest

# Enviar imagem para o ECR
echo -e "${YELLOW}Enviando imagem para o ECR (isso pode demorar)...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/news-recommender-api:latest

echo -e "${GREEN}Imagem enviada com sucesso para o ECR.${NC}"
echo -e "${GREEN}Caminho da imagem: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/news-recommender-api:latest${NC}"

echo -e "${YELLOW}==== PRÓXIMOS PASSOS ====${NC}"
echo -e "${YELLOW}1. Acesse o console AWS App Runner: https://console.aws.amazon.com/apprunner${NC}"
echo -e "${YELLOW}2. Crie um novo serviço usando a imagem ECR:${NC}"
echo -e "${YELLOW}   - URL da imagem: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/news-recommender-api:latest${NC}"
echo -e "${YELLOW}   - Porta: 8000${NC}"
echo -e "${YELLOW}   - Variáveis de ambiente necessárias:${NC}"
echo -e "${YELLOW}     * ENV=prod${NC}"
echo -e "${YELLOW}     * MLFLOW_TRACKING_URI=http://ec2-3-93-215-88.compute-1.amazonaws.com:5000/${NC}"
echo -e "${YELLOW}     * AWS_ACCESS_KEY_ID=[sua chave]${NC}"
echo -e "${YELLOW}     * AWS_SECRET_ACCESS_KEY=[sua chave secreta]${NC}"
echo -e "${YELLOW}3. Aguarde o deploy ser concluído e obtenha a URL da aplicação${NC}"

echo -e "${GREEN}Script concluído com sucesso!${NC}"