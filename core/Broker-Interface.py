import fxcmpy as forex_broker
import json

API_TOKEN = json.load(open(r"C:\Users\Melgiri\PycharmProjects\ForEx\settings\settings.json"))["FXCM"]["REST API Token"]


def main():
	client = forex_broker.fxcmpy(config_file=r"C:\Users\Melgiri\PycharmProjects\ForEx\settings\fxcm.cfg")
	df = client.get_accounts_summary()
	print(df["balance"])


if __name__ == '__main__':
	main()
