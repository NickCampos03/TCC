# 📚 TCC - Detecção de ataques em aplicações web hospedadas na nuvem pública utilizando aprendizado de máquina aplicado à análise de logs. 

Este repositório contém o Trabalho de Conclusão de Curso (TCC) do curso de Bacharelado em Ciência da Computação do IFSP - Campus Salto, desenvolvido em 2026.

---

## 🎯 Objetivo

O objetivo deste trabalho é desenvolver uma **machine learning** capaz de analisar **ataques cibernéticos** a partir de logs extraidos de uma aplicação web de sistema de gerenciamento de biblioteca desenvolvida no ano de 2025
---

## 🏗️ Arquitetura do Sistema

O sistema é dividido em:

* **Frontend**: Interface web desenvolvida em React
* **Backend**: API REST desenvolvida em Spring Boot
* **Banco de Dados**: PostgreSQL
* **Camada de Segurança**: Filtro de requisições + análise de logs
* **Módulo de IA**: Classificação de requisições maliciosas

---

## ⚙️ Tecnologias Utilizadas

* Java (Spring Boot)
* React.js
* PostgreSQL
* Python (Machine Learning) e script de automação
* Git e GitHub

---

## 🚀 Funcionalidades

### 📖 Sistema de Biblioteca

* Gerenciamento de usuários
* Gerenciamento de livros
* Gerenciamento de Generos do sistema
* Empréstimos e devoluções
* Controle de disponibilidade

### 🔐 Segurança

* Captura de requisições HTTP
* Registro de logs detalhados
* Detecção de:
  * Brute Force
  * SQL Injection
  * XSS (Cross-Site Scripting)

### 🤖 Inteligência Artificial

* Análise de logs
* Classificação de requisições
* Identificação de padrões maliciosos

---

## 🗂️ Estrutura do Projeto

```bash
TCC/
├── Aplicacao SGB/       # Sistema principal (Spring + React) 
├── TCC/                 # Documentação e materiais do trabalho, como scripts e documentção
└── README.md
```

---

## 🧪 Exemplos de Ataques Testados

### SQL Injection

```sql
' OR '1'='1
```

### XSS

```html
<script>alert('XSS')</script>
```

---

## 📊 Coleta de Dados

Os logs são capturados através de um filtro no backend (`RequestLoggingFilter`), contendo:

* IP do usuário
* Método HTTP
* Endpoint acessado
* Corpo da requisição

Esses dados são utilizados para treinar e testar o modelo de Machine Learning.

---

## ▶️ Como Executar o Projeto

### 🔹 Backend (Spring Boot)

```bash
cd Aplicacao SGB
git submodule update --init --recursive
cd SGB---Backend/sgb-api/sgb-api
mvn spring-boot:run
```

### 🔹 Frontend (React)

```bash
cd Aplicacao SGB
git submodule update --init --recursive
cd SGB---Frontend/sgb-web/sgb-web/web
npm install
npm run dev
```

---

## 📌 Status do Projeto

🚧 Em desenvolvimento

---

## 👨‍💻 Autores

Gabriel Cândido
Nicolas Campos
Alunos de Bacharelado de Ciência da Computação - IFSP Salto

---
 
