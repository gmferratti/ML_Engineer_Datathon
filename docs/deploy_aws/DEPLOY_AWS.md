# Deploy do Sistema de Recomendação na AWS App Runner

Este guia simplificado mostra como fazer o deploy do sistema de recomendação usando AWS App Runner - a maneira mais simples e direta de colocar containers na AWS sem precisar configurar infraestrutura complexa.

## Pré-requisitos

1. Uma conta AWS
2. AWS CLI instalado e configurado em sua máquina
3. Docker instalado em sua máquina
4. Correção do erro "No module named 'configs'" (instruções abaixo)

Existe um script **deploy-to-aws.sh** na raiz do projeto que automatiza as etapas abaixo.

## Etapa 1: Obter o ID da sua conta AWS

Existem várias maneiras de obter seu AWS Account ID:

### Opção 1: Usando o AWS CLI
```bash
aws sts get-caller-identity --query Account --output text
```

### Opção 2: No Console AWS
1. Faça login no console AWS
2. Clique no nome do seu usuário no canto superior direito
3. Seu Account ID aparecerá no dropdown

## Etapa 2: Criar um repositório ECR e enviar a imagem Docker

```bash
# Substitua 123456789012 pelo seu Account ID
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# Criar repositório ECR
aws ecr create-repository --repository-name news-recommender-api

# Autenticar no ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Construir a imagem Docker
docker build -t news-recommender-api .

# Tagear a imagem para o ECR
docker tag news-recommender-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/news-recommender-api:latest

# Enviar a imagem para o ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/news-recommender-api:latest
```

## Etapa 3: Deploy usando AWS App Runner pelo Console

1. Acesse o [Console AWS App Runner](https://console.aws.amazon.com/apprunner)
2. Clique em **Create service**

### Configurar fonte
1. Em **Source**, selecione **Container registry**
2. Em **Provider**, selecione **Amazon ECR**
3. Em **Container image URI**, selecione a imagem que acabou de enviar:
   `$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/news-recommender-api:latest`
4. Em **Deployment settings**, selecione **Manual**
5. Clique em **Next**

### Configurar serviço
1. Dê um nome ao serviço: `news-recommender-api`
2. Em **Port**, insira `8000`
3. Em **CPU**, selecione `1 vCPU`
4. Em **Memory**, selecione `2 GB`
5. Em **Environment variables**, adicione:
   - `ENV`: `prod`
   - `MLFLOW_TRACKING_URI`: URL do seu MLflow remoto
   - `AWS_ACCESS_KEY_ID`: Sua chave de acesso AWS
   - `AWS_SECRET_ACCESS_KEY`: Sua chave secreta AWS
   - `AWS_DEFAULT_REGION`: Região AWS (ex: us-east-1)
6. Clique em **Next**

### Revisar e criar
1. Revise as configurações
2. Clique em **Create & deploy**

O App Runner começará a provisionar seu serviço, o que pode levar alguns minutos. Quando estiver pronto, você terá um endpoint HTTPS para acessar a API.

## Etapa 4: Testar a API

Depois que o serviço estiver em execução, você pode testar a API usando curl ou um navegador:

```bash
# Verificar se a API está funcionando
curl https://[seu-app-runner-url]/health

# Fazer uma requisição de recomendação
curl -X POST https://[seu-app-runner-url]/predict \
  -H "Content-Type: application/json" \
  -d '{"userId": "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297", "max_results": 5}'
```

## Dicas e Solução de Problemas

### Permissões para ECR
Se o App Runner não conseguir extrair a imagem, verifique se você configurou corretamente as permissões de acesso ao ECR:

1. No console ECR, selecione o repositório que você criou
2. Vá para a guia **Permissions**
3. Adicione uma política que conceda acesso ao serviço App Runner para extrair imagens

### Logs
Para verificar os logs em caso de problemas:

1. No console App Runner, selecione seu serviço
2. Clique na guia **Logs**
3. Você verá logs de construção e implantação, bem como logs do aplicativo

### Atualização da Aplicação
Para atualizar sua aplicação:

1. Faça alterações em seu código
2. Reconstrua e envie a imagem Docker para o ECR
3. No console App Runner, selecione seu serviço e clique em **Deploy**

### Escalonamento
App Runner escala automaticamente com base na carga. Você pode configurar:

1. No console App Runner, selecione seu serviço
2. Clique na guia **Configuration**
3. Em **Auto scaling**, defina:
   - **Min size**: 1 (mínimo de instâncias sempre rodando)
   - **Max size**: 5 (máximo para escalar durante carga alta)

### Monitoramento
O App Runner integra-se automaticamente ao CloudWatch para métricas e logs:

1. No console App Runner, selecione seu serviço
2. Clique na guia **Metrics** para ver o desempenho