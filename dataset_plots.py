import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_json('train.json')
test_data = pd.read_json('test.json')

plt.figure(figsize=(12, 8))
plot = sns.countplot(x='marca', data=train_data)
plt.xticks(rotation=90)

for p in plot.patches:
    height = p.get_height()
    plot.annotate(f'{height}',
                  (p.get_x() + p.get_width() / 2., height),
                  ha='center', va='center',
                  xytext=(0, 5), textcoords='offset points')

plt.xlabel("Manufacturer", labelpad=20)
plt.ylabel("Numărul de mașini (train)")
plt.title("Numărul de mașini per marca")
plt.show()


plt.figure(figsize=(12, 8))
plot = sns.countplot(x='marca', data=test_data)
plt.xticks(rotation=90)

for p in plot.patches:
    height = p.get_height()
    plot.annotate(f'{height}',
                  (p.get_x() + p.get_width() / 2., height),
                  ha='center', va='center',
                  xytext=(0, 5), textcoords='offset points')

plt.xlabel("Manufacturer", labelpad=20)
plt.ylabel("Numărul de mașini (test)")
plt.title("Numărul de mașini per marca")
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(train_data['pret'], kde=True)
plt.title('Distribuția prețurilor mașinilor (train.json)')
plt.xlabel('Preț')
plt.ylabel('Numărul de mașini')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='an', data=train_data)
plt.title('Numărul de mașini pe an de fabricație (train.josn)')
plt.xlabel('An de fabricație')
plt.ylabel('Numărul de mașini')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='an', data=test_data)
plt.title('Numărul de mașini pe an de fabricație (test.json)')
plt.xlabel('An de fabricație')
plt.ylabel('Numărul de mașini')
plt.xticks(rotation=45)
plt.show()


def plot_addons_frequency_per_brand(data):
    addons_per_brand = pd.DataFrame()

    for brand in data['marca'].unique():
        brand_data = data[data['marca'] == brand]
        addons_list = [addon for sublist in brand_data['addons'] for addon in sublist]
        addons_count = pd.Series(addons_list).value_counts().head(5)
        temp_df = pd.DataFrame({'Marca': brand, 'Addon': addons_count.index, 'Frecvență': addons_count.values})
        addons_per_brand = pd.concat([addons_per_brand, temp_df])

    plt.figure(figsize=(15, 10))
    sns.barplot(x='Addon', y='Frecvență', hue='Marca', data=addons_per_brand)
    plt.title('Top 5 dotări per marca')
    plt.xticks(rotation=45)
    plt.show()

def plot_year_distribution_per_brand(data):
    plt.figure(figsize=(15, 10))
    sns.countplot(x='an', hue='marca', data=data)
    plt.title('Distribuția anului de fabricație per marcă')
    plt.xticks(rotation=45)
    plt.show()


def plot_mileage_vs_price_per_brand(data):
    custom_palette = sns.color_palette("husl", len(data['marca'].unique()))

    plt.figure(figsize=(16, 10))
    ax = sns.scatterplot(x='km', y='pret', hue='marca', data=data, alpha=0.6, palette=custom_palette)

    plt.title('Numărul de kilometri și preț per marcă')
    plt.xlabel('Număr kilometri')
    plt.ylabel('Preț')

    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Marca')
    for text in lgd.get_texts():
        text.set_fontsize('small')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


plot_addons_frequency_per_brand(train_data)
plot_addons_frequency_per_brand(test_data)
plot_year_distribution_per_brand(train_data)
plot_year_distribution_per_brand(test_data)
plot_mileage_vs_price_per_brand(train_data)

