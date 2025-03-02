#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "======================================================="
echo "   Sistema de Recomendação de Notícias - Inicialização"
echo "======================================================="
echo -e "${NC}"

# Função de ajuda
function show_help {
    echo -e "${CYAN}Uso: $0 [ambiente] [modo] [opções]${NC}"
    echo
    echo -e "Ambientes:"
    echo -e "  ${GREEN}dev${NC}    Ambiente de desenvolvimento (padrão)"
    echo -e "  ${GREEN}prod${NC}   Ambiente de produção"
    echo
    echo -e "Modos:"
    echo -e "  ${GREEN}full${NC}   Inicia todos os serviços (padrão)"
    echo -e "  ${GREEN}api${NC}    Inicia apenas a API"
    echo -e "  ${GREEN}mlflow${NC} Inicia apenas o MLflow (só em dev)"
    echo
    echo -e "Opções:"
    echo -e "  ${GREEN}rebuild${NC} Reconstrói as imagens antes de iniciar"
    echo -e "  ${GREEN}logs${NC}    Exibe logs após iniciar"
    echo -e "  ${GREEN}help${NC}    Exibe esta ajuda"
    echo
    echo -e "Exemplos:"
    echo -e "  ${YELLOW}$0${NC}                   # Inicia em dev com todos os serviços"
    echo -e "  ${YELLOW}$0 prod api${NC}          # Inicia apenas a API em produção"
    echo -e "  ${YELLOW}$0 dev api rebuild${NC}   # Reconstrói e inicia a API em dev"
    echo
    exit 0
}

# Processar argumentos para exibir ajuda
for arg in "$@"; do
    if [ "$arg" == "help" ]; then
        show_help
    fi
done

# Verificar dependências
echo -e "${YELLOW}Verificando dependências...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker não encontrado. Por favor, instale o Docker e tente novamente.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose não encontrado. Por favor, instale o Docker Compose e tente novamente.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Todas as dependências encontradas.${NC}"

# Verificar se existem os diretórios necessários
echo -e "${YELLOW}Verificando diretórios de dados...${NC}"
mkdir -p data
mkdir -p mlruns

# Definir ambiente e modo
ENV="dev"
MODE="full" 
REBUILD="false"
SHOW_LOGS="false"

# Processar argumentos para definir ENV, MODE e opções
for arg in "$@"; do
    case $arg in
        "dev"|"prod")
            ENV=$arg
            ;;
        "api"|"mlflow"|"full")
            MODE=$arg
            ;;
        "rebuild")
            REBUILD="true"
            ;;
        "logs")
            SHOW_LOGS="true"
            ;;
    esac
done

echo -e "${YELLOW}Ambiente: ${CYAN}$ENV${NC}"
echo -e "${YELLOW}Modo: ${CYAN}$MODE${NC}"

# Configurar variáveis de ambiente
if [ "$ENV" == "prod" ]; then
    export ENV="prod"
    export COMPOSE_PROFILES=""
    
    # Verificar se as credenciais AWS estão definidas para modo prod
    if [ "$ENV" == "prod" ] && [ -z "$AWS_ACCESS_KEY_ID" ]; then
        echo -e "${YELLOW}⚠️ Executando em produção sem AWS_ACCESS_KEY_ID definido${NC}"
        echo -e "${YELLOW}⚠️ O acesso a dados no S3 pode falhar${NC}"
    fi
else
    export ENV="dev" 
    export COMPOSE_PROFILES="dev"
    export MLFLOW_LOCAL_DIR="./mlruns"
fi

# Reconstruir se solicitado
if [ "$REBUILD" == "true" ]; then
    echo -e "${YELLOW}Reconstruindo as imagens Docker...${NC}"
    docker-compose down
    docker-compose build --no-cache
fi

# Iniciar serviços de acordo com o modo
case $MODE in
    "api")
        echo -e "${YELLOW}Iniciando apenas a API em modo ${CYAN}$ENV${NC}...${NC}"
        docker-compose up -d api
        ;;
    "mlflow")
        if [ "$ENV" == "prod" ]; then
            echo -e "${RED}MLflow standalone não é suportado em produção.${NC}"
            echo -e "${YELLOW}Em produção, a API se conectará diretamente ao MLflow em:${NC}"
            grep "MLFLOW_TRACKING_URI" src/configs/prod.yaml | sed 's/.*: //'
            exit 1
        fi
        echo -e "${YELLOW}Iniciando apenas MLflow...${NC}"
        docker-compose --profile dev up -d mlflow
        ;;
    *)
        echo -e "${YELLOW}Iniciando todos os serviços em modo ${CYAN}$ENV${NC}...${NC}"
        if [ "$ENV" == "dev" ]; then
            docker-compose --profile dev up -d
        else
            docker-compose up -d api
        fi
        ;;
esac

# Verificar se os serviços iniciaram corretamente
echo -e "${YELLOW}Verificando status dos serviços...${NC}"
sleep 5

if docker-compose ps | grep -q "api" && docker-compose ps | grep "api" | grep -q "Up"; then
    echo -e "${GREEN}✓ API iniciada com sucesso.${NC}"
    echo -e "${BLUE}API disponível em: http://localhost:8000${NC}"
    echo -e "${BLUE}Documentação da API: http://localhost:8000/docs${NC}"
else
    echo -e "${RED}✗ API não iniciou corretamente. Verifique os logs: docker-compose logs api${NC}"
fi

if [ "$ENV" == "dev" ] && [ "$MODE" != "api" ] && docker-compose ps | grep -q "mlflow" && docker-compose ps | grep "mlflow" | grep -q "Up"; then
    echo -e "${GREEN}✓ MLflow iniciado com sucesso.${NC}"
    echo -e "${BLUE}MLflow UI disponível em: http://localhost:5001${NC}"
else
    if [ "$ENV" == "dev" ] && [ "$MODE" != "api" ]; then
        echo -e "${RED}✗ MLflow não iniciou corretamente. Verifique os logs: docker-compose logs mlflow${NC}"
    fi
fi

echo -e "${YELLOW}Ambiente atual: ${CYAN}$ENV${NC}"
if [ "$ENV" == "prod" ]; then
    echo -e "${YELLOW}Usando MLflow remoto em:${NC}"
    grep "MLFLOW_TRACKING_URI" src/configs/prod.yaml | sed 's/.*: //'
else
    echo -e "${YELLOW}Usando MLflow local em: http://localhost:5001${NC}"
fi

# Exibir logs se solicitado, forçando toda a saída em verde
if [ "$SHOW_LOGS" == "true" ]; then
    echo -e "${YELLOW}Exibindo logs (Ctrl+C para parar)...${NC}"
    docker-compose logs -f --ansi=never | sed -r "s/\x1B\[[0-9;]*[a-zA-Z]//g" | awk '{print "\033[0;32m" $0 "\033[0m"}'
else
    echo -e "${YELLOW}Para visualizar logs: docker-compose logs -f${NC}"
fi

echo -e "${YELLOW}Para parar os serviços: docker-compose down${NC}"
echo -e "${GREEN}Sistema inicializado!${NC}"
