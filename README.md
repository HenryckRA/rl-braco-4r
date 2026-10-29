# Ensinando um braço robótico de 4 juntas a alcançar alvos com Aprendizado por Reforço

Projeto da disciplina **GBC063 — Inteligência Artificial** (UFU/FACOM), Prof. Marcelo Albertini.

**Dupla:** Henrique Rodrigues Alves · Lucas Lara Carvalho

Em vez de programar o caminho do braço ponto a ponto, um agente de RL aprende por tentativa e erro **quanta força (torque) aplicar em cada motor** para levar a ponta até um alvo no espaço, chegando com o ângulo pedido (por exemplo, por cima). O agente não conhece as equações do robô: só observa o estado, age e recebe recompensa. A física é simulada no [MuJoCo](https://mujoco.org).

## O braço

4 juntas rotativas: **1 base** (gira em torno do eixo vertical), **2 ombro**, **3 cotovelo** e **4 punho** (sobem e descem o braço). O alvo tem 4 requisitos (x, y, z e o ângulo de chegada) para 4 juntas, então não há redundância.

## Formulação (MDP)

| | Definição |
|---|---|
| **Estado** `sₜ ∈ ℝ²¹` | `cos qᵢ, sin qᵢ` (8), velocidades `q̇ᵢ` (4), posição do alvo (3), `cos φ*, sin φ*` (2), erro de posição `Δp` (3), erro de ângulo `Δφ` (1) |
| **Ação** `aₜ ∈ [−1, 1]⁴` | Torque normalizado de cada motor: `τᵢ = τmax,ᵢ · aᵢ` (contínua); o agente decide a cada 20 ms |
| **Recompensa** | `rₜ = −d − w·|Δφ| − λ‖aₜ‖² + Rsuc − Rlim` |
| **Sucesso** | `d < 2 cm`, `|Δφ| < 10°` e braço praticamente parado → bônus e fim da tentativa |
| **Violação** | Junta acima da velocidade máxima ou ponta abaixo do chão → penalidade e fim da tentativa |
| **Duração** | Até 200 decisões (4 s) |

O alvo é gerado a partir de uma pose aleatória do próprio braço, o que garante que ele é alcançável com o ângulo pedido.

Algoritmos previstos: **PPO** e **SAC** (Stable-Baselines3).

## Estrutura

```
├── docs/
│   └── etapa1/              # apresentação da Etapa 1 (PDF)
├── src/braco4/
│   ├── braco4.xml           # modelo do braço no MuJoCo (elos, juntas, motores)
│   └── ambiente.py          # ambiente Gymnasium Braco4-v0 (estado, ação, recompensa)
├── scripts/
│   ├── treinar.py           # treino PPO/SAC
│   └── visualizar.py        # abre o visualizador 3D do MuJoCo
└── tests/
    └── test_ambiente.py     # checagem da API Gymnasium e sanidade da física
```

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pytest                                            # valida o ambiente
python scripts/visualizar.py                      # braço com ações aleatórias (3D)
python scripts/treinar.py --algo sac              # treino (próximas etapas)
python scripts/visualizar.py resultados/sac_braco4.zip
```

## Entregas

- [x] **Etapa 1 (29/10/2026):** descrição do problema: ambiente/estado, ações e recompensas ([`docs/etapa1`](docs/etapa1))
- [ ] Etapa 2: _a definir pelo professor_
- [ ] Etapa 3: _a definir pelo professor_
