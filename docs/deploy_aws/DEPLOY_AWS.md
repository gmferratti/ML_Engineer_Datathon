# Deploy do Sistema de Recomendação na AWS

## Abordagens de Deploy

Durante o desenvolvimento deste projeto, exploramos duas estratégias de deploy na AWS:

1. **Deploy via Amazon ECR e AWS App Runner** (Abordagem Inicial)
2. **Deploy Direto em EC2** (Solução Implementada)

## Limitações do Ambiente de Laboratório AWS

No ambiente de laboratório fornecido, encontramos restrições significativas que impactaram nossa estratégia de deploy:

- Acesso limitado aos serviços de Container Registry (ECR)
- Permissões restritas para criação de recursos de container
- Dificuldades na configuração de serviços gerenciados como App Runner

## Abordagem 1: ECR e App Runner (Planejamento Original)

Esta abordagem foi inicialmente planejada como a solução ideal para deploy de containers na AWS, aproveitando os benefícios do AWS App Runner.

### Etapas Planejadas
1. Obter o ID da conta AWS
2. Criar repositório ECR
3. Autenticar e enviar imagem Docker
4. Configurar App Runner
5. Definir variáveis de ambiente
6. Realizar deploy do serviço

### Exemplo de Comandos
```bash
# Criar repositório ECR
aws ecr create-repository --repository-name news-recommender-api

# Autenticar e enviar imagem
aws ecr get-login-password | docker login
docker build -t news-recommender-api .
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/news-recommender-api:latest
```

## Abordagem 2: Deploy Direto em EC2 (Solução Implementada)

Devido às limitações do ambiente de laboratório, migramos para um método de deploy direto em instância EC2.

### Benefícios desta Abordagem
- Menor complexidade de configuração
- Maior flexibilidade no ambiente de laboratório
- Controle direto sobre a infraestrutura
- Facilidade de depuração e monitoramento

### Script de Deploy

O script `deploy-to-aws.sh` automatiza o processo de:
- Criação de instância EC2
- Configuração do ambiente
- Instalação do Docker
- Deploy do container

### Exemplo de Uso

```bash
# Preparar o script
chmod +x deploy-to-aws.sh

# Executar o deploy
./deploy-to-aws.sh
```

## Recomendações para Ambientes Reais

Em um ambiente de produção com acesso completo, recomendamos:
- Utilizar serviços gerenciados como App Runner ou ECS
- Implementar infraestrutura como código (IaC)
- Configurar pipelines de CI/CD
- Usar serviços como Amazon SageMaker
