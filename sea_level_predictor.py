{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPWlynTaIzYxVPZyp9+miMF",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/velaz7/Sea-Level-Predictor/blob/main/sea_level_predictor.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "t2g-rsK63Q1N"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from scipy.stats import linregress"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('/content/epa-sea-level.csv')\n",
        "df.head()\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 223
        },
        "id": "FpyjqBHr4cOE",
        "outputId": "65ba5771-b7ce-4f5e-d4d1-e19ff6dac120"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Year  CSIRO Adjusted Sea Level  Lower Error Bound  Upper Error Bound  \\\n",
              "0  1880                  0.000000          -0.952756           0.952756   \n",
              "1  1881                  0.220472          -0.732283           1.173228   \n",
              "2  1882                 -0.440945          -1.346457           0.464567   \n",
              "3  1883                 -0.232283          -1.129921           0.665354   \n",
              "4  1884                  0.590551          -0.283465           1.464567   \n",
              "\n",
              "   NOAA Adjusted Sea Level  \n",
              "0                      NaN  \n",
              "1                      NaN  \n",
              "2                      NaN  \n",
              "3                      NaN  \n",
              "4                      NaN  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d406352b-bea5-4153-b028-73e2cf4aebb1\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Year</th>\n",
              "      <th>CSIRO Adjusted Sea Level</th>\n",
              "      <th>Lower Error Bound</th>\n",
              "      <th>Upper Error Bound</th>\n",
              "      <th>NOAA Adjusted Sea Level</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1880</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>-0.952756</td>\n",
              "      <td>0.952756</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1881</td>\n",
              "      <td>0.220472</td>\n",
              "      <td>-0.732283</td>\n",
              "      <td>1.173228</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1882</td>\n",
              "      <td>-0.440945</td>\n",
              "      <td>-1.346457</td>\n",
              "      <td>0.464567</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1883</td>\n",
              "      <td>-0.232283</td>\n",
              "      <td>-1.129921</td>\n",
              "      <td>0.665354</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1884</td>\n",
              "      <td>0.590551</td>\n",
              "      <td>-0.283465</td>\n",
              "      <td>1.464567</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d406352b-bea5-4153-b028-73e2cf4aebb1')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-d406352b-bea5-4153-b028-73e2cf4aebb1 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-d406352b-bea5-4153-b028-73e2cf4aebb1');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-33594e30-0245-4a7f-8a59-683c750e5220\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-33594e30-0245-4a7f-8a59-683c750e5220')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-33594e30-0245-4a7f-8a59-683c750e5220 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 134,\n  \"fields\": [\n    {\n      \"column\": \"Year\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 38,\n        \"min\": 1880,\n        \"max\": 2013,\n        \"num_unique_values\": 134,\n        \"samples\": [\n          2007,\n          1946,\n          1984\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"CSIRO Adjusted Sea Level\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.485691970383905,\n        \"min\": -0.440944881,\n        \"max\": 9.326771644,\n        \"num_unique_values\": 128,\n        \"samples\": [\n          2.826771651,\n          1.988188974,\n          1.338582676\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Lower Error Bound\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.663781329340837,\n        \"min\": -1.346456692,\n        \"max\": 8.992125975,\n        \"num_unique_values\": 129,\n        \"samples\": [\n          1.83070866,\n          1.472440943,\n          0.574803149\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Upper Error Bound\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2.3125811058145285,\n        \"min\": 0.464566929,\n        \"max\": 9.661417313,\n        \"num_unique_values\": 127,\n        \"samples\": [\n          1.783464565,\n          7.330708654,\n          4.708661413\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"NOAA Adjusted Sea Level\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.6910380781791361,\n        \"min\": 6.297493046,\n        \"max\": 8.546648227,\n        \"num_unique_values\": 21,\n        \"samples\": [\n          6.297493046,\n          8.122972567,\n          7.90736541\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Lin_Reg = linregress(x=df['Year'], y=df['CSIRO Adjusted Sea Level'])\n",
        "\n"
      ],
      "metadata": {
        "id": "Fw4a_U4u9BO_"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y2=Lin_Reg.intercept + Lin_Reg.slope * df['Year']\n",
        "sns.scatterplot(x='Year',y='CSIRO Adjusted Sea Level',data=df,label='CSIRO Sea Level')\n",
        "sns.lineplot(x='Year',y= y2,data=df,color='r',label='fitted line')\n",
        "plt.title('Sea Level by CSIRO')\n",
        "plt.xlabel('Year')\n",
        "plt.ylabel('Sea Level (inches)')"
      ],
      "metadata": {
        "id": "iHtPsgVf9KcI",
        "outputId": "a74b1336-23c8-4d1f-f8f1-eb09f6ca1879",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        }
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, 'Sea Level (inches)')"
            ]
          },
          "metadata": {},
          "execution_count": 11
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_last = df[df['Year']>=2000]\n",
        "\n",
        "Lin_Reg_Last = linregress(x=df_last['Year'], y=df_last['CSIRO Adjusted Sea Level'])\n",
        "print(Lin_Reg_Last)\n"
      ],
      "metadata": {
        "id": "SLHmHnkZ97Fw",
        "outputId": "e379daf6-0857-46cb-aacb-d9207da56662",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "LinregressResult(slope=0.1664272733318682, intercept=-325.7934668059649, rvalue=0.9762875716140618, pvalue=2.4388064141618245e-09, stderr=0.010652933111541163, intercept_stderr=21.375153425608215)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x1= df_last['Year']\n",
        "y1=df_last['CSIRO Adjusted Sea Level']\n",
        "x2=np.arange(2000,2051)\n",
        "y2=Lin_Reg_Last.intercept + Lin_Reg_Last.slope*x2\n",
        "plt.scatter(x1, y1, label='CSIRO sea  Level')\n",
        "plt.plot(x2, y2, label='fitted line',color='b')\n",
        "plt.xlabel('Year')\n",
        "plt.ylabel('Sea Level (inches)')\n",
        "plt.title('Sea Level by CSIRO Until Year 2050')\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "qyurxIpV-O8P",
        "outputId": "b153a048-300f-4664-d333-28be9dca077b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        }
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjIAAAHHCAYAAACle7JuAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABrk0lEQVR4nO3de3zO9f/H8ce12Xk25rhpzuSYU+SQs5yKlJxy1kElEkWqb6IklUqKQpFDOYTSAZVDJDnPIWdtyDHGZhvbbO/fH5/frsyG2a7t2uF5v912q8/h+lyv62O2p8/7ZDPGGERERERyIBdnFyAiIiKSXgoyIiIikmMpyIiIiEiOpSAjIiIiOZaCjIiIiORYCjIiIiKSYynIiIiISI6lICMiIiI5loKMiIiI5FgKMiI5ROnSpenXr99NzwkLC8Nms/Hee+9lTVHiULNmzcJmsxEWFmbf16xZM5o1a+a0mkSyOwUZcardu3fzyCOPUKpUKTw9PSlRogT33XcfkydPzvJa1q5di81m45tvvsny986pjhw5wsCBAylbtiyenp74+fnRqFEjJk2axOXLl+3nxcXFMWnSJGrVqoWfnx8FChSgatWqPPnkk+zfv99+XtIv8q1bt9r3vf7669hsNvuXm5sbpUuXZsiQIVy8eDHVuuLj4/noo4+oW7cu+fPnx9fXl7p16/LRRx8RHx+fps/WrFkzqlWrluqxc+fOYbPZeP3119N0reu99dZbfPvtt+l67fU2btyIi4sLo0aNSvX4hAkTsNls/Pjjjw55v/RatWoVAwYMoGLFinh7e1O2bFkef/xxTp06ler5f/zxB/feey/e3t4UL16cIUOGEBUVleycpL+zqX39+eef6bqm5Dz5nF2A5F1//PEHzZs3p2TJkjzxxBMUL16c48eP8+effzJp0iQGDx7s7BLlJn788Ue6dOmCh4cHffr0oVq1asTFxfH777/z4osv8tdffzFt2jQAOnfuzPLly+nRowdPPPEE8fHx7N+/nx9++IGGDRtSqVKlW77f1KlT8fX1JTo6mlWrVjF58mS2b9/O77//nuy86Oho7r//fn777TceeOAB+vXrh4uLCytWrOC5555jyZIl/Pjjj/j4+GTKfUmLt956i0ceeYROnTol29+7d2+6d++Oh4dHmq/VoEEDBg4cyMSJE+nVqxdVq1a1Hzt69Chjx46lS5cu3H///Y4qP11GjhxJeHg4Xbp0oUKFCvz99998/PHH/PDDD4SEhFC8eHH7uSEhIbRs2ZLKlSvz/vvv888///Dee+9x6NAhli9fnuLaQ4YMoW7dusn2lS9fPtn27V5TchAj4iTt27c3RYoUMRcuXEhx7MyZM1lez5o1awxgFi1alOXvnRalSpUyffv2vek5oaGhBjDvvvtuptby999/G19fX1OpUiVz8uTJFMcPHTpkPvzwQ2OMMZs3bzaAGTduXIrzrl69as6dO2ffnjlzpgHMli1b7PtGjx5tAPPvv/8me223bt0MYDZt2pRs/5NPPmkAM3ny5BTv9/HHHxvAPPXUU7f8jE2bNjVVq1ZN9di///5rADN69OhbXic1Pj4+t/yzvLaOpk2b3vScixcvmsDAQNOoUSOTmJho39+hQwfj7++f6p9RZoiOjr7hsd9++80kJCSk2AeYV155Jdn+du3amcDAQBMREWHfN336dAOYlStX2vfdzt/ZtF5Tch41LYnTHDlyhKpVq1KgQIEUx4oWLZpi39y5c6lTpw5eXl4EBATQvXt3jh8/nuyc9evX06VLF0qWLImHhwfBwcE8//zzyZo5MurixYsMHTqU4OBgPDw8KF++PBMmTCAxMRGwmjUCAgLo379/itdGRkbi6enJCy+8YN8XGxvL6NGjKV++vL3mESNGEBsbm6E6P/jgA0qVKoWXlxdNmzZlz5499mMzZ87EZrOxY8eOFK976623cHV15cSJEze89jvvvENUVBSff/45gYGBKY6XL1+e5557DrD+nAEaNWqU4jxXV1cKFSp0258NoHHjxsmuD/DPP//w+eef06JFC5599tkUrxk0aBDNmzdnxowZ/PPPP+l63xtJagI7fPgw/fr1o0CBAvj7+9O/f39iYmLs59lsNqKjo/nyyy/tzSBJfZ9S6yOTFv7+/kyaNIkNGzYwY8YMAJYuXcr333/P22+/TWBgIImJiXz44YdUrVoVT09PihUrxsCBA7lw4UKya3333Xfcf//9BAUF4eHhQbly5XjjjTdISEhIdl5S09u2bdto0qQJ3t7evPzyyzessUmTJri4uKTYFxAQwL59++z7IiMj+eWXX+jVqxd+fn72/X369MHX15eFCxemev1Lly5x9erVVI+l95qSMyjIiNOUKlWKbdu2JfsFeyPjxo2jT58+VKhQgffff5+hQ4eyatUqmjRpkqyfxKJFi4iJieHpp59m8uTJtGnThsmTJ9OnTx+H1BwTE0PTpk2ZO3cuffr04aOPPqJRo0aMGjWKYcOGAeDm5sZDDz3Et99+S1xcXLLXf/vtt8TGxtK9e3cAEhMT6dixI++99x4dOnRg8uTJdOrUiQ8++IBu3bqlu87Zs2fz0UcfMWjQIEaNGsWePXto0aIFZ86cAeCRRx7By8uLefPmpXjtvHnzaNasGSVKlLjh9b///nvKli1Lw4YNb1lLqVKl7Ne90S+a9Ej6ZV+wYEH7vuXLl5OQkHDTP+8+ffpw9epVVqxY4bBartW1a1cuXbrE+PHj6dq1K7NmzWLMmDH243PmzMHDw4PGjRszZ84c5syZw8CBAzP8vknNRyNHjuTvv//mueeeo2HDhvZrDxw4kBdffNHeh6l///7MmzePNm3aJOs3NGvWLHx9fRk2bBiTJk2iTp06vPbaa7z00ksp3vP8+fO0a9eOmjVr8uGHH9K8efPbqjkqKoqoqCgKFy5s37d7926uXr3K3Xffnexcd3d3atasmWr47t+/P35+fnh6etK8efNkfazSe03JQZz9SEjyrp9//tm4uroaV1dX06BBAzNixAizcuVKExcXl+y8sLAw4+rqmqJpYvfu3SZfvnzJ9sfExKR4n/HjxxubzWaOHj1603rS8pj6jTfeMD4+PubgwYPJ9r/00kvG1dXVHDt2zBhjzMqVKw1gvv/++2TntW/f3pQtW9a+PWfOHOPi4mLWr1+f7LxPP/3UAGbDhg32fbfTtOTl5WX++ecf+/5NmzYZwDz//PP2fT169DBBQUHJHvdv377dAGbmzJk3fI+IiAgDmAcffPCmtSRJTEw0TZs2NYApVqyY6dGjh/nkk09S/fO4WdPSgQMHzL///mvCwsLMF198Yby8vEyRIkWSNWcMHTrUAGbHjh03rCfpMw4bNuymdd9u01JSnQMGDEh27kMPPWQKFSqUbN+NmpaSPn9oaGiyOm7VtJQkLCzM+Pj4mICAAOPm5mZ2795tjDFm/fr1BjDz5s1Ldv6KFStS7E/t79DAgQONt7e3uXLlSrK6APPpp5+mqbbUvPHGGwYwq1atsu9btGiRAcy6detSnN+lSxdTvHhx+/aGDRtM586dzeeff26+++47M378eFOoUCHj6elptm/fnq5rSs6jJzLiNPfddx8bN26kY8eO7Ny5k3feeYc2bdpQokQJli1bZj9vyZIlJCYm0rVrV86dO2f/Kl68OBUqVGDNmjX2c728vOz/Hx0dzblz52jYsCHGGIf8q2vRokU0btyYggULJqulVatWJCQksG7dOgBatGhB4cKFWbBggf21Fy5c4Jdffkn2pGXRokVUrlyZSpUqJbteixYtAJJ9ttvRqVOnZE9U6tWrxz333MNPP/1k39enTx9OnjyZ7D3mzZuHl5cXnTt3vuG1IyMjAcifP3+aarHZbKxcuZI333yTggUL8vXXXzNo0CBKlSpFt27dbjjy6Hp33nknRYoUoXTp0gwYMIDy5cuzfPlyvL297edcunTplrUlHUv6HI721FNPJdtu3Lgx58+fz7T3u1apUqUYPXo04eHhDBs2zD7qatGiRfj7+3Pfffcl+z6rU6cOvr6+N/w7dOnSJc6dO0fjxo2JiYlJNsIMwMPDI9Um1LRYt24dY8aMoWvXrvbvd8DeDJxah2dPT89kzcQNGzbkm2++YcCAAXTs2JGXXnqJP//8E5vNlmwU1+1cU3IejVoSp6pbty5LliwhLi6OnTt3snTpUj744AMeeeQRQkJCqFKlCocOHcIYQ4UKFVK9hpubm/3/jx07xmuvvcayZctStP1HRERkuN5Dhw6xa9cuihQpkurxs2fPApAvXz46d+7MV199RWxsLB4eHixZsoT4+PhkQebQoUPs27fvlte7Xandq4oVKybrC3DfffcRGBjIvHnzaNmyJYmJiXz99dc8+OCDNw0CSX0MkkJDWnh4ePDKK6/wyiuvcOrUKX777TcmTZrEwoULcXNzY+7cube8xuLFi/Hz8+Pff//lo48+IjQ0NNkvXfgvpNystrSEnbSy2Wwp9pUsWTLZdlLT14ULF5L1z8gsSaN3rm1GOXToEBEREan2PYPk32d//fUXr776KqtXr04Rvq7/O1SiRAnc3d1vu8b9+/fz0EMPUa1aNXufniRJf6ap9RG7cuVKij/z65UvX54HH3yQJUuWkJCQgKura4avKdmbgoxkC+7u7tStW5e6detSsWJF+vfvz6JFixg9ejSJiYnYbDaWL1+Oq6tritf6+voCkJCQwH333Ud4eDgjR46kUqVK+Pj4cOLECfr162fvjJsRiYmJ3HfffYwYMSLV4xUrVrT/f/fu3fnss89Yvnw5nTp1YuHChVSqVIkaNWoku1716tV5//33U71ecHBwhmu+EVdXVx599FGmT5/OlClT2LBhAydPnqRXr143fZ2fnx9BQUFp6tuUmsDAQLp3707nzp2pWrUqCxcuZNasWeTLd/MfR02aNLH3pejQoQPVq1enZ8+ebNu2zd6JtHLlygDs2rWLmjVrpnqdXbt2AVClSpWbvt/N/qWe1HnX09MzxbHUvkcBjDE3fb/MlJiYSNGiRVPtEwXYg/TFixdp2rQpfn5+jB07lnLlyuHp6cn27dsZOXJkir9D6QkAx48fp3Xr1vj7+/PTTz+lCJRJncdTm1/m1KlTBAUF3fI9goODiYuLIzo6Gj8/P4dcU7IvBRnJdpL+JZn0Q6dcuXIYYyhTpkyyoHC93bt3c/DgQb788stknT1/+eUXh9VWrlw5oqKiaNWq1S3PbdKkCYGBgSxYsIB7772X1atX88orr6S43s6dO2nZsmWq/7pPr0OHDqXYd/DgQUqXLp1sX58+fZg4cSLff/89y5cvp0iRIrRp0+aW13/ggQeYNm0aGzdupEGDBumq0c3NjbvuuotDhw7ZmwrTytfXl9GjR9O/f38WLlxo7zzdrl07XF1dmTNnzg07/M6ePZt8+fLRtm3bm75HqVKlWL16NZcvX07xC/vAgQP2c9LDkX/WaVGuXDl+/fVXGjVqdNPwsXbtWs6fP8+SJUto0qSJfX9oaKhD6jh//jytW7cmNjaWVatWpTrirVq1auTLl4+tW7fStWtX+/64uDhCQkKS7buRv//+G09PT/s/chxxTcm+1EdGnGbNmjWp/is1qR/HnXfeCcDDDz+Mq6srY8aMSXG+MYbz588D//1L+NpzjDFMmjTJYTV37dqVjRs3snLlyhTHLl68mGxUjouLC4888gjff/89c+bM4erVqylGInXt2pUTJ04wffr0FNe7fPky0dHR6arz22+/TTZ8evPmzWzatIl27dolO++uu+7irrvuYsaMGSxevJju3bvf8skIwIgRI/Dx8eHxxx+3j4S61pEjR+z3/dChQxw7dizFORcvXmTjxo0ULFjwhk1rN9OzZ0/uuOMOJkyYYN8XHBxM//79+fXXX5k6dWqK13z66aesXr2axx57jDvuuOOm12/fvj3x8fF89tlnyfYnJiYydepU3N3dadmy5W3XDeDj45PmvkGO0LVrVxISEnjjjTdSHLt69aq9ltT+DsXFxTFlypQM1xAdHU379u05ceIEP/300w2biv39/WnVqhVz585N1kQ4Z84coqKi6NKli33fv//+m+L1O3fuZNmyZbRu3dr+pO52rik5j57IiNMMHjyYmJgYHnroISpVqkRcXBx//PEHCxYsoHTp0vZOhOXKlePNN99k1KhRhIWF0alTJ/Lnz09oaChLly7lySef5IUXXqBSpUqUK1eOF154gRMnTuDn58fixYtT9JW5lcWLF6fo1AjQt29fXnzxRZYtW2afMbZOnTpER0eze/duvvnmG8LCwpINJe3WrRuTJ09m9OjRVK9e3d70kaR3794sXLiQp556ijVr1tCoUSMSEhLYv38/CxcuZOXKlSmGjKZF+fLluffee3n66aeJjY3lww8/pFChQqk2ifXp08c+r82tmpWSlCtXjq+++opu3bpRuXLlZDP7/vHHHyxatMg+N8rOnTt59NFHadeuHY0bNyYgIIATJ07w5ZdfcvLkST788MMbNsfcjJubG8899xwvvvgiK1assD9h+eCDD9i/fz/PPPNMsv0rV67ku+++o2nTpkycOPGW1+/QoQOtW7fm+eefZ/PmzTRs2JCYmBiWLVvGhg0bePPNN9MVwADq1KnDr7/+yvvvv09QUBBlypThnnvuSde10qJp06YMHDiQ8ePHExISQuvWrXFzc+PQoUMsWrSISZMm8cgjj9CwYUMKFixI3759GTJkCDabjTlz5jikWaxnz55s3ryZAQMGsG/fvmRzx/j6+iab5XjcuHE0bNiQpk2b8uSTT/LPP/8wceJEWrdunexJWrdu3fDy8qJhw4YULVqUvXv3Mm3aNLy9vXn77beTvX9aryk5kJNGS4mY5cuXmwEDBphKlSoZX19f4+7ubsqXL28GDx6c6sy+ixcvNvfee6/x8fExPj4+plKlSmbQoEHmwIED9nP27t1rWrVqZXx9fU3hwoXNE088YXbu3HnLIcXG/Df8+kZfSUOkL126ZEaNGmXKly9v3N3dTeHChU3Dhg3Ne++9l2LoeGJiogkODjaAefPNN1N937i4ODNhwgRTtWpV4+HhYQoWLGjq1KljxowZk2wW0tud2XfixIkmODjYeHh4mMaNG5udO3em+ppTp04ZV1dXU7FixZteOzUHDx40TzzxhCldurRxd3c3+fPnN40aNTKTJ0+2D9U9c+aMefvtt03Tpk1NYGCgyZcvnylYsKBp0aKF+eabb5Jd73Zm9jXGGgru7++fYnhybGys+eCDD0ydOnWMj4+P8fb2NrVr1zYffvhhij+jm7ly5Yp5/fXXTaVKlYyHh4fx8fEx9evXN3Pnzk1x7o3qTG1I9f79+02TJk2Ml5eXAex/rhkdfm3MzacRmDZtmqlTp47x8vIy+fPnN9WrVzcjRoxINvPvhg0bTP369Y2Xl5cJCgqyT4sAmDVr1iSr60bD01NTqlSpG/7dKlWqVIrz169fbxo2bGg8PT1NkSJFzKBBg0xkZGSycyZNmmTq1atnAgICTL58+UxgYKDp1auXOXToUKo1pOWakvPYjHFiDzQRcbpz584RGBjIa6+9xv/+9z9nlyMiclvUR0Ykj5s1axYJCQn07t3b2aWIiNw29ZERyaNWr17N3r17GTduHJ06dUoxoklEJCdQ05JIHtWsWTP++OMPGjVqxNy5c2+6tpKISHalICMiIiI5lvrIiIiISI6lICMiIiI5Vq7v7JuYmMjJkyfJnz9/lk8LLiIiIuljjOHSpUsEBQXZZ2lOTa4PMidPnszUhfdEREQk8xw/fvymS4rk+iCTtLLq8ePH8fPzc3I1IiIikhaRkZEEBwenWCH9erk+yCQ1J/n5+SnIiIiI5DC36haizr4iIiKSYynIiIiISI6lICMiIiI5Vq7vI5NWCQkJxMfHO7sMEdzc3HB1dXV2GSIiOUKeDzLGGE6fPs3FixedXYqIXYECBShevLjmPhIRuYU8H2SSQkzRokXx9vbWLw5xKmMMMTExnD17FoDAwEAnVyQikr3l6SCTkJBgDzGFChVydjkiAHh5eQFw9uxZihYtqmYmEZGbyNOdfZP6xHh7ezu5EpHkkr4n1W9LROTm8nSQSaLmJMlu9D0pIpI2CjIiIiKSYynIiDhBv3796NSpk7PLEBHJ8RRkcqjTp08zePBgypYti4eHB8HBwXTo0IFVq1bZz9m5cycdO3akaNGieHp6Urp0abp162YfERMWFobNZiMkJCTZdtJXQEAATZs2Zf369SnePzw8nKFDh1KqVCnc3d0JCgpiwIABHDt2LEs+f0asXbsWm82mIfciIrmAgowDJCQaNh45z3chJ9h45DwJiSZT3y8sLIw6deqwevVq3n33XXbv3s2KFSto3rw5gwYNAuDff/+lZcuWBAQEsHLlSvbt28fMmTMJCgoiOjr6ptf/9ddfOXXqFOvWrSMoKIgHHniAM2fO2I+Hh4dTv359fv31Vz799FMOHz7M/PnzOXz4MHXr1uXvv//O1M8vIiLZQ0IC/Pijk4swuVxERIQBTERERIpjly9fNnv37jWXL19O9/WX7z5p6r/1qyk18gf7V/23fjXLd5/MSNk31a5dO1OiRAkTFRWV4tiFCxeMMcYsXbrU5MuXz8THx9/wOqGhoQYwO3bsSHXbGGN27dplAPPdd9/Z9z311FPGx8fHnDp1Ktn1YmJiTIkSJUzbtm1v+J5hYWHmgQceMAUKFDDe3t6mSpUq5scff7Qf3717t2nbtq3x8fExRYsWNb169TL//vuv/fjy5ctNo0aNjL+/vwkICDD333+/OXz48A3fLzVr1qwxgP1eXe/KlStm+PDhJigoyHh7e5t69eqZNWvWGGOs7ydPT0/z008/JXvNkiVLjK+vr4mOjjbGGHPs2DHTpUsX4+/vbwoWLGg6duxoQkND7ef37dvXPPjggzes0RHfmyIimenUKWNatjQGjPn6a8df/2a/v6+lJzIZsGLPKZ6eu51TEVeS7T8dcYWn525nxZ5TDn/P8PBwVqxYwaBBg/Dx8UlxvECBAgAUL16cq1evsnTpUoxJ3xOiy5cvM3v2bADc3d0BSExMZP78+fTs2ZPixYsnO9/Ly4tnnnmGlStXEh4enuo1Bw0aRGxsLOvWrWP37t1MmDABX19fAC5evEiLFi2oVasWW7duZcWKFZw5c4auXbvaXx8dHc2wYcPYunUrq1atwsXFhYceeojExMR0fcbUPPvss2zcuJH58+eza9cuunTpQtu2bTl06BB+fn488MADfPXVV8leM2/ePDp16oS3tzfx8fG0adOG/Pnzs379ejZs2ICvry9t27YlLi7OYXWKiDjLqlVQs6b1X29vcOCP4NuWpyfEy4iERMOY7/eSWkQwgA0Y8/1e7qtSHFcXxw2lPXz4MMYYKlWqdNPz6tevz8svv8yjjz7KU089Rb169WjRogV9+vShWLFiN31tw4YNcXFxISYmBmMMderUoWXLloDVZHXx4kUqV66c6msrV66MMYbDhw9Tr169FMePHTtG586dqV69OgBly5a1H/v444+pVasWb731ln3fF198QXBwMAcPHqRixYp07tw52fW++OILihQpwt69e6lWrdpNP1daHDt2jJkzZ3Ls2DGCgoIAeOGFF1ixYgUzZ87krbfeomfPnvTu3ZuYmBi8vb2JjIzkxx9/ZOnSpQAsWLCAxMREZsyYYR9GPXPmTAoUKMDatWtp3bp1husUEXGGhAQYMwbefBOMgWrVYMECqFLFeTXpiUw6bQ4NT/Ek5loGOBVxhc2hqT+ZSK/beboybtw4Tp8+zaeffkrVqlX59NNPqVSpErt3777p6xYsWMCOHTtYvHgx5cuXZ9asWbi5uaW7jmsNGTKEN998k0aNGjF69Gh27dplP7Zz507WrFmDr6+v/SspsB05cgSAQ4cO0aNHD8qWLYufnx+lS5cGcFgn4927d5OQkEDFihWT1fHbb7/Za2jfvj1ubm4sW7YMgMWLF+Pn50erVq3sn+Pw4cPkz5/f/vqAgACuXLliv4aISE5z8iS0bAlvvGGFmMcfh02bnBtiQE9k0u3spRuHmPScl1YVKlTAZrOxf//+NJ1fqFAhunTpQpcuXXjrrbeoVasW7733Hl9++eUNXxMcHEyFChWoUKECV69e5aGHHmLPnj14eHhQpEgRChQowL59+1J97b59+7DZbJQvXz7V448//jht2rThxx9/5Oeff2b8+PFMnDiRwYMHExUVRYcOHZgwYUKK1yWtOdShQwdKlSrF9OnTCQoKIjExkWrVqjmsySYqKgpXV1e2bduWYmmApCYwd3d3HnnkEb766iu6d+/OV199Rbdu3ciXL5/9GnXq1GHevHkprl+kSBGH1CkikpVWroReveDcOfD1hc8+g0cfdXZVFj2RSaei+T0del5aBQQE0KZNGz755JNURx/dbEixu7s75cqVu+WopWs98sgj5MuXjylTpgDg4uJC165d+eqrrzh9+nSycy9fvsyUKVNo06YNAQEBN7xmcHAwTz31FEuWLGH48OFMnz4dgNq1a/PXX39RunRpypcvn+zLx8eH8+fPc+DAAV599VVatmxJ5cqVuXDhQpo/S1rUqlWLhIQEzp49m6KGa/sE9ezZkxUrVvDXX3+xevVqevbsaT9Wu3ZtDh06RNGiRVNcw9/f36H1iohkpqtXYdQoaNvWCjE1asC2bdknxICCTLrVKxNAoL8nN+r9YgMC/T2pV+bGv9DT65NPPiEhIYF69eqxePFiDh06xL59+/joo49o0KABAD/88AO9evXihx9+4ODBgxw4cID33nuPn376iQcffDDN72Wz2RgyZAhvv/02MTExALz11lsUL16c++67j+XLl3P8+HHWrVtHmzZtiI+P55NPPrnh9YYOHcrKlSsJDQ1l+/btrFmzxt7fZtCgQYSHh9OjRw+2bNnCkSNHWLlyJf379ychIYGCBQtSqFAhpk2bxuHDh1m9ejXDhg1L933cvXs3ISEh9q+dO3dSsWJFevbsSZ8+fViyZAmhoaFs3ryZ8ePH8+M1YwybNGlC8eLF6dmzJ2XKlOGee+6xH+vZsyeFCxfmwQcfZP369YSGhrJ27VqGDBnCP//8k+56RUSy0vHj0KwZvP22tf300/Dnn1CxolPLSkFBJp1cXWyM7mA1DF4fZpK2R3eo4tCOvknKli3L9u3bad68OcOHD6datWrcd999rFq1iqlTpwJQpUoVvL29GT58ODVr1qR+/fosXLiQGTNm0Lt379t6v759+xIfH8/HH38MWM1Vf/75J82bN2fgwIGUK1eOrl27Uq5cObZs2ZKsA+/1EhISGDRoEJUrV6Zt27ZUrFjR/rQnKCiIDRs2kJCQQOvWralevTpDhw6lQIECuLi44OLiwvz589m2bRvVqlXj+eef5913303nXbTCSK1atexfderUAayOuX369GH48OHceeeddOrUiS1btlCyZEn7a202Gz169GDnzp3JnsaAteDjunXrKFmyJA8//DCVK1fmscce48qVK/j5+aW7XhGRrPLjj9aopA0bwM/P6tA7ZQp4OraRwSFsJr29NnOIyMhI/P39iYiISPFL5MqVK4SGhlKmTBk80/mns2LPKcZ8vzdZx99Af09Gd6hC22qBGapd8i5HfG+KiNyu+HirKWniRGu7dm1YuBDKlcv6Wm72+/ta6uybQW2rBXJfleJsDg3n7KUrFM1vNSdlxpMYERGRzBIWBt27WyORAIYMgXfeAQ8Pp5Z1SwoyDuDqYqNBuULOLkNERCRdvv0W+veHixehQAH44gt46CEnF5VG6iMjIiKSR8XGwtChVmi5eBHq1YMdO3JOiAEFGRERkTzp77+hUSOYNMnaHj4c1q+H/59nNMdQ05KIiEge88038NhjEBkJAQEwaxZ06ODsqtJHT2RERETyiCtXYNAg6NLFCjENG0JISM4NMaAgIyIikiccOgQNGljzwQCMHAlr10JwsFPLyjA1LYmIiORyX38NTz4J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NYsWLSJfvnzky5ePli1bAlC4cGFGjx5tv47uteN/Rt9zzz38888/xMbG2q+TGfc5TwcZYwzPPvssS5cuZfXq1ZQpUybZ8Tp16uDm5saqVavs+w4cOMCxY8do0KABAA0aNGD37t2cPXvWfs4vv/yCn58fVapUsZ9z7TWSzkm6Rm7niPucFg0aNGDdunXEx8fb9/3yyy/ceeedFCxYMOMfJJtz1H3+66+/aN68OX379mXcuHEp3kffz5nz/ZyYmAhg/6Gf1+8zOOZeL168mJ07dxISEkJISIi9+W79+vUMGjQI0L3OrO/pkJAQChYsaF9QMtPuc4a6CudwTz/9tPH39zdr165NNlwsJibGfs5TTz1lSpYsaVavXm22bt1qGjRoYBo0aGA/njT8unXr1iYkJMSsWLHCFClSJNXh1y+++KLZt2+f+eSTT/LU0D5H3GdjjDl06JDZsWOHGThwoKlYsaLZsWOH2bFjh32U0sWLF02xYsVM7969zZ49e8z8+fONt7d3nhl+7Yj7vHv3blOkSBHTq1evZNc4e/as/Rx9P2f8Pv/444/miy++MLt37zahoaHmhx9+MJUrVzaNGjWyn5PX77MxjvvZca01a9bccPh1Xr3XjrjPy5YtM9OnTze7d+82hw4dMlOmTDHe3t7mtddes5+TWfc5TwcZINWvmTNn2s+5fPmyeeaZZ0zBggWNt7e3eeihh8ypU6eSXScsLMy0a9fOeHl5mcKFC5vhw4eb+Pj4ZOesWbPG1KxZ07i7u5uyZcsme4/czlH3uWnTpqleJzQ01H7Ozp07zb333ms8PDxMiRIlzNtvv51Fn9L5HHGfR48eneo1SpUqley99P2csfu8evVq06BBA+Pv7288PT1NhQoVzMiRI5P9cjUmb99nYxz3s+NaqQWZpP159V474j4vX77c1KxZ0/j6+hofHx9To0YN8+mnn5qEhIRk75UZ99n2/x9CREREJMfJ031kREREJGdTkBEREZEcS0FGREREciwFGREREcmxFGREREQkx1KQERERkRxLQUZERERyLAUZERERybEUZETE6YwxtGrVijZt2qQ4NmXKFAoUKMA///zjhMpEJLtTkBERp7PZbMycOZNNmzbx2Wef2feHhoYyYsQIJk+enGw1Xke4dnFREcm5FGREJFsIDg5m0qRJvPDCC4SGhmKM4bHHHqN169bUqlWLdu3a4evrS7Fixejduzfnzp2zv3bFihXce++9FChQgEKFCvHAAw9w5MgR+/GwsDBsNhsLFiygadOmeHp6Mm/ePGd8TBFxMK21JCLZSqdOnYiIiODhhx/mjTfe4K+//qJq1ao8/vjj9OnTh8uXLzNy5EiuXr3K6tWrAVi8eDE2m4277rqLqKgoXnvtNcLCwggJCcHFxYWwsDDKlClD6dKlmThxIrVq1cLT05PAwEAnf1oRySgFGRHJVs6ePUvVqlUJDw9n8eLF7Nmzh/Xr17Ny5Ur7Of/88w/BwcEcOHCAihUrprjGuXPnKFKkCLt376ZatWr2IPPhhx/y3HPPZeXHEZFMpqYlEclWihYtysCBA6lcuTKdOnVi586drFmzBl9fX/tXpUqVAOzNR4cOHaJHjx6ULVsWPz8/SpcuDcCxY8eSXfvuu+/O0s8iIpkvn7MLEBG5Xr58+ciXz/rxFBUVRYcOHZgwYUKK85Kahjp06ECpUqWYPn06QUFBJCYmUq1aNeLi4pKd7+Pjk/nFi0iWUpARkWytdu3aLF68mNKlS9vDzbXOnz/PgQMHmD59Oo0bNwbg999/z+oyRcRJ1LQkItnaoEGDCA8Pp0ePHmzZsoUjR46wcuVK+vfvT0JCAgULFqRQoUJMmzaNw4cPs3r1aoYNG+bsskUkiyjIiEi2FhQUxIYNG0hISKB169ZUr16doUOHUqBAAVxcXHBxcWH+/Pls27aNatWq8fzzz/Puu+86u2wRySIatSQiIiI5lp7IiIiISI6lICMiIiI5loKMiIiI5FgKMiIiIpJjKciIiIhIjqUgIyIiIjmWgoyIiIjkWAoyIiIikmMpyIiIiEiOpSAjIiIiOZaCjIiIiORYCjIiIiKSY/0freX5qKcLAmwAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "val_2050 = Lin_Reg_Last.intercept + Lin_Reg_Last.slope*2050\n",
        "print(val_2050)"
      ],
      "metadata": {
        "id": "X4_CBQhH_CBh",
        "outputId": "2a4f858b-586b-4fbd-e1a6-ed5f52d826ac",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "15.382443524364874\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Rncm4cF1_PZ2"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}