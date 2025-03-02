#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}  Deploy do Sistema de Recomendação em EC2 ${NC}"
echo -e "${BLUE}===========================================${NC}"

# Verificar dependências
echo -e "${YELLOW}Verificando dependências...${NC}"
command -v aws >/dev/null 2>&1 || { echo -e "${RED}AWS CLI não encontrado. Instale-o primeiro.${NC}"; exit 1; }

# Configurações
AWS_REGION="us-east-1"  # Altere conforme necessário
KEY_NAME="vockey"       # Chave SSH que você usará
INSTANCE_TYPE="t2.medium"  # Aumentei para t2.medium para suportar o modelo
SECURITY_GROUP_NAME="news-recommender-sg"
INSTANCE_NAME="news-recommender-instance"
AMI_ID="ami-0fc5d935ebf8bc3bc" # Ubuntu 22.04 para us-east-1
DOCKER_IMAGE="mauricioarauujo1/news-recommender-api:latest"

# Função para limpar recursos em caso de erro
cleanup() {
    echo -e "${RED}Erro no deployment. Limpando recursos...${NC}"
    if [ -n "$INSTANCE_ID" ]; then
        aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $AWS_REGION
    fi
    
    # Remover setup_and_run.sh se existir
    [ -f setup_and_run.sh ] && rm setup_and_run.sh
}

# Configurar trap para capturar erros
trap cleanup ERR

# Verificar se já existe uma instância com esse nome
echo -e "${YELLOW}Verificando instâncias existentes...${NC}"
EXISTING_INSTANCE=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text \
    --region $AWS_REGION)

if [ -n "$EXISTING_INSTANCE" ]; then
    echo -e "${GREEN}Instância $INSTANCE_NAME já existe (ID: $EXISTING_INSTANCE). Usando a instância existente.${NC}"
    INSTANCE_ID=$EXISTING_INSTANCE
    
    # Obter o IP público da instância existente
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $AWS_REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
else
    # Verificar se o grupo de segurança já existe
    echo -e "${YELLOW}Verificando grupo de segurança...${NC}"
    SG_CHECK=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --region $AWS_REGION 2>&1)
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}Grupo de segurança $SECURITY_GROUP_NAME já existe. Usando-o.${NC}"
        SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --region $AWS_REGION --query 'SecurityGroups[0].GroupId' --output text)
    else
        # Criar grupo de segurança
        echo -e "${YELLOW}Criando grupo de segurança...${NC}"
        SECURITY_GROUP_ID=$(aws ec2 create-security-group \
          --group-name $SECURITY_GROUP_NAME \
          --description "Security group for News Recommender API" \
          --region $AWS_REGION \
          --query 'GroupId' \
          --output text)
        
        # Configurar regras de segurança
        echo -e "${YELLOW}Configurando regras de segurança...${NC}"
        aws ec2 authorize-security-group-ingress \
          --group-id $SECURITY_GROUP_ID \
          --protocol tcp \
          --port 22 \
          --cidr 0.0.0.0/0 \
          --region $AWS_REGION

        aws ec2 authorize-security-group-ingress \
          --group-id $SECURITY_GROUP_ID \
          --protocol tcp \
          --port 8000 \
          --cidr 0.0.0.0/0 \
          --region $AWS_REGION
    fi

    # Criar a instância EC2
    echo -e "${YELLOW}Criando instância EC2...${NC}"
    INSTANCE_ID=$(aws ec2 run-instances \
      --image-id $AMI_ID \
      --count 1 \
      --instance-type $INSTANCE_TYPE \
      --key-name $KEY_NAME \
      --security-group-ids $SECURITY_GROUP_ID \
      --region $AWS_REGION \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
      --user-data "$(cat <<EOF
#!/bin/bash
# Script de inicialização para preparar a instância
apt update -y
apt upgrade -y
EOF
)" \
      --query 'Instances[0].InstanceId' \
      --output text)

    echo -e "${GREEN}Instância $INSTANCE_ID criada com sucesso.${NC}"

    # Aguardar a instância estar pronta
    echo -e "${YELLOW}Aguardando instância ficar disponível...${NC}"
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $AWS_REGION
    aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID --region $AWS_REGION

    # Obter o IP público
    PUBLIC_IP=$(aws ec2 describe-instances \
      --instance-ids $INSTANCE_ID \
      --region $AWS_REGION \
      --query 'Reservations[0].Instances[0].PublicIpAddress' \
      --output text)
fi

echo -e "${GREEN}Instância iniciada com sucesso! IP público: $PUBLIC_IP${NC}"

# Criar script de preparação e execução
echo -e "${YELLOW}Criando script de preparação...${NC}"
cat > setup_and_run.sh << 'EOF'
#!/bin/bash
# Script de configuração e execução do serviço

# Atualizar sistema
sudo apt update -y
sudo apt upgrade -y

# Instalar dependências necessárias
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Adicionar chave GPG oficial do Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Adicionar repositório do Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Atualizar novamente para reconhecer o novo repositório
sudo apt update -y

# Instalar Docker
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Iniciar e habilitar Docker
sudo systemctl start docker
sudo systemctl enable docker

# Adicionar usuário atual ao grupo docker
sudo usermod -aG docker $USER

# Criar diretório para aplicação
mkdir -p ~/news-recommender
cd ~/news-recommender

# Puxar a imagem Docker
sudo docker pull mauricioarauujo1/news-recommender-api:latest

# Iniciar o container
export ENV=prod
ENV=prod sudo docker compose up -d

echo "Deployment concluído! API disponível em http://$(hostname -I | awk '{print $1}'):8000"
EOF

# Tornar o script executável
chmod +x setup_and_run.sh

# Enviar os arquivos para a instância
echo -e "${YELLOW}Enviando arquivos para a instância...${NC}"
scp -o StrictHostKeyChecking=no -i ~/.ssh/labsuser.pem docker-compose.yml setup_and_run.sh ubuntu@$PUBLIC_IP:~/news-recommender/

# Executar o script de instalação
echo -e "${YELLOW}Executando script de instalação...${NC}"
ssh -o StrictHostKeyChecking=no -i ~/.ssh/labsuser.pem ubuntu@$PUBLIC_IP "cd ~/news-recommender && chmod +x setup_and_run.sh && ./setup_and_run.sh"

# Remover o script local após o envio
rm setup_and_run.sh

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Deployment concluído com sucesso!${NC}"
echo -e "${GREEN}API disponível em: http://$PUBLIC_IP:8000${NC}"
echo -e "${GREEN}Para verificar o status: ssh -i ~/.ssh/labsuser.pem ubuntu@$PUBLIC_IP 'sudo docker logs news-recommender-api'${NC}"
echo -e "${GREEN}===========================================${NC}"

# Limpeza do trap de erro
trap - ERR