import gradio as gr
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def generate_random_numbers():
    start = -10
    end = 10
    x_11 = np.random.randint(start, end)
    x_12 = np.random.randint(start, end)
    w_11 = np.random.randint(start, end)
    w_12 = np.random.randint(start, end)
    w_21 = np.random.randint(start, end)
    w_22 = np.random.randint(start, end)
    b1_1 = np.random.randint(start, end)
    b2_1 = np.random.randint(start, end)
    w_1 = np.random.randint(start, end)
    w_2 = np.random.randint(start, end)
    b1_2 = np.random.randint(start, end)
    return (x_11, x_12, w_11, w_12, w_21, w_22, b1_1, b2_1, w_1, w_2, b1_2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def func_activation(z, func):
    if func == "Sigmoid":
        return sigmoid(z)
    elif func == "ReLU":
        return relu(z)
    elif func == "Tanh":
        return tanh(z)
    elif func == "Leaky ReLU":
        return leaky_relu(z)

def error_func(y_real, y_pred):
    return (y_pred - y_real) ** 2

def create_graph(x_11, x_12, w_11, w_12, w_21, w_22, b1_1, b2_1, w_1, w_2, b1_2, a_hidden1, a_hidden2, a_output):
    G = nx.DiGraph()
   
    G.add_node("Entrada 1", value=x_11)
    G.add_node("Entrada 2", value=x_12)
    G.add_node("Oculto 1", value=a_hidden1)
    G.add_node("Oculto 2", value=a_hidden2)
    G.add_node("Saída", value=a_output)
    
    G.add_edge("Entrada 1", "Oculto 1", weight=w_11)
    G.add_edge("Entrada 1", "Oculto 2", weight=w_12)
    G.add_edge("Entrada 2", "Oculto 1", weight=w_21)
    G.add_edge("Entrada 2", "Oculto 2", weight=w_22)
    G.add_edge("Oculto 1", "Saída", weight=w_1)
    G.add_edge("Oculto 2", "Saída", weight=w_2)

    pos = nx.spring_layout(G, k=1.7, iterations=50)

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=3000, font_size=10, arrows=True
    )

    node_labels = {node: f"\n\n({data['value']:.2f})" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    img = Image.open(buf)
    return img

def calc_forward_pass(x_11, x_12, w_11, w_12, w_21, w_22, b1_1, b2_1, w_1, w_2, b1_2, func, y_real):
    # Entrada
    x = np.array([[x_11], [x_12]])

    # Pesos e Bias (Camada 1)
    w1 = np.array([[w_11, w_12], 
                    [w_21, w_22]])
    b1 = np.array([[b1_1], 
                    [b2_1]])

    # Pesos e Bias (Camada 2)
    w2 = np.array([[w_1, w_2]])
    b2 = np.array([[b1_2]])

    # Forward Pass
    h_linear = w1 @ x + b1
    h = func_activation(h_linear, func)
    z = w2 @ h + b2
    y = func_activation(z, func)
    
    # Saída
    y_pred = y.item()

    # Cálculo do Erro
    mse = error_func(y_real, y_pred)

    graph_img = create_graph(x_11, x_12, w_11, w_12, w_21, w_22, b1_1, b2_1, w_1, w_2, b1_2, h[0][0], h[1][0], y_pred)

    return f"Saída da camada oculta:\n{h}\nValor z (entrada da saída): {z.item():.3f}\nSaída final y: {y.item():.6f}\nErro Quadrático Médio (MSE): {mse:.6f}", graph_img

with gr.Blocks() as demo:
    gr.Markdown("## Simulador de Rede Neural Simples")
    
    with gr.Row():
        gr.Markdown("### Primeira Camada")
        x_11 = gr.Number(label="x_11", value=generate_random_numbers()[0])
        x_12 = gr.Number(label="x_12", value=generate_random_numbers()[1])
        w_11 = gr.Number(label="w_11", value=generate_random_numbers()[2])
        w_12 = gr.Number(label="w_12", value=generate_random_numbers()[3])
        w_21 = gr.Number(label="w_21", value=generate_random_numbers()[4])
        w_22 = gr.Number(label="w_22", value=generate_random_numbers()[5])
        b1_1 = gr.Number(label="b1", value=generate_random_numbers()[6])
        b2_1 = gr.Number(label="b2", value=generate_random_numbers()[7])
    
    with gr.Row():
        gr.Markdown("### Segunda Camada")
        w_1 = gr.Number(label="w1", value=generate_random_numbers()[8])
        w_2 = gr.Number(label="w2", value=generate_random_numbers()[9])
        b1_2 = gr.Number(label="b1", value=generate_random_numbers()[10])
    
    func = gr.Dropdown(["Sigmoid", "ReLU", "Tanh", "Leaky ReLU"], value="Sigmoid", label="Função (f)")
    y_real = gr.Number(label="y_real", value=0.5)
    
    btn = gr.Button(value="Calcular")
    btn.click(
        calc_forward_pass,
        inputs=[x_11, x_12, w_11, w_12, w_21, w_22, b1_1, b2_1, w_1, w_2, b1_2, func, y_real],
        outputs=[
            gr.Textbox(label="Resultados"),
            gr.Image(label="Gráfico da Rede Neural", type="pil")
        ]
    )

demo.launch()