import matplotlib.pyplot as plt

def plot_pie_chart(data, title='Pie Chart'):
    labels = list(data.keys())
    sizes = list(data.values())

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    # plt.title(title)
    # plt.show()
    plt.tight_layout()
    plt.savefig(title + ".svg", dpi=400)

# Example usage

# ILU(1), q(0.5) GPU
# data = {
#     'triang' : 0.803,
#     'SpMV' : 1.4e-3,
#     'GS' : 0.143,
#     'factor' : 0.06
# }

# ILU(7), q(0.5) 
data = {
    'triang' : 5.574,
    'SpMV' : 2.82e-3,
    'GS' : 0.118,
    'factor' : 1.747,
}

plot_pie_chart(data, title='ILU(1)-q(0.5)-GPU')