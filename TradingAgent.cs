
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using System.Net.Http;
using Newtonsoft.Json;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
#endregion


namespace NinjaTrader.NinjaScript.Strategies
{
	public class TradingAgent : Strategy
	{	
		#region Properties

		[NinjaScriptProperty]
		[Display(Name = "Trade Amount", GroupName = "Trading Agent Parameters", Order = 2)]
		public int tradeAmount
		{ get; set; }

		#endregion

		private int numBar;
		private int sequenceLength = 10;
		private int totalBarsInSession = 1380;
		
		// Initial price data for normalizing
		private bool initCollected = false;
		private double initOpen;
		private double initHigh;
		private double initLow;
		private double initClose;

		public class RequestData
        {
            public List<double> progress = new List<double>();
            public List<double> open = new List<double>();
            public List<double> high = new List<double>();
            public List<double> low = new List<double>();
            public List<double> close = new List<double>();
            public List<double> volume = new List<double>();
            public int contracts;
            public double entryPrice;
            public double balance;
		}

		private RequestData data = new RequestData();

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Enter the description for your new custom Strategy here.";
				Name										= "TradingAgent";
				Calculate									= Calculate.OnBarClose;
				EntriesPerDirection							= 1;
				EntryHandling								= EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy				= true;
				ExitOnSessionCloseSeconds					= 30;
				IsFillLimitOnTouch							= false;
				MaximumBarsLookBack							= MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution							= OrderFillResolution.Standard;
				Slippage									= 0;
				StartBehavior								= StartBehavior.ImmediatelySubmit;
				TimeInForce									= TimeInForce.Gtc;
				TraceOrders									= false;
				RealtimeErrorHandling						= RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling							= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade							= 0;
				IsInstantiatedOnEachOptimizationIteration	= true;
				tradeAmount									= 1;
			}
		}
		
		
		protected override async void OnBarUpdate()
		{
		    if (State == State.Historical)
                return;	

			if (CurrentBars[0] > 0 && !initCollected)
			{
				initCollected = true;

				initOpen = Open[0];
				initHigh = High[0];
				initLow = Low[0];
				initClose = Close[0];
			}
			
			data.progress.Add(((double)Bars.BarsSinceNewTradingDay / totalBarsInSession) * 100);

			numBar = CurrentBar;
			if(numBar % sequenceLength == 0 && numBar != 0)
			{
				CollectData();
				SendHttpRequest(Newtonsoft.Json.JsonConvert.SerializeObject(data));
				ClearData();
			}
		}
		
		private void CollectData()
		{			
			for (int i = 0; i < sequenceLength; i++)
			{
				data.open.Add(Open[i] - initOpen);
				data.high.Add(High[i] - initHigh);
				data.low.Add(Low[i] - initLow);
				data.close.Add(Close[i] - initClose);
				data.volume.Add(Volume[i]);
			}
			data.contracts = Position.Quantity;
			data.entryPrice = Position.AveragePrice;
			data.balance = GetAccountBalance();
		}
		
		private void ClearData()
		{
			data.progress.Clear();
			data.open.Clear();
			data.high.Clear();
			data.low.Clear();
			data.close.Clear();
			data.volume.Clear();
		}
		
		private double GetAccountBalance()
		{
			Account a = Account.All.First(t => t.Name == "Playback101");
			double value = a.Get(AccountItem.CashValue, Currency.UsDollar);
			return value;
		}
		
		private async void SendHttpRequest(string json)
        {
            using (var httpClient = new HttpClient())
            {
                try
                {
                    var url = "http://localhost:8000/predict";
                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    var response = await httpClient.PostAsync(url, content);

                    if (response.IsSuccessStatusCode)
                    {
                        var responseContent = await response.Content.ReadAsStringAsync();
                        Print("POST request sent successfully. Response: " + responseContent);
						
		                dynamic parsedResponse = Newtonsoft.Json.JsonConvert.DeserializeObject(responseContent);
		                double prediction = parsedResponse.prediction;
						
						ExitPosition();
						if (prediction > 0)
						{
							EnterLong(tradeAmount);
							Print("Entered a long position with amount: " + tradeAmount);
						}
						else if (prediction < 0)
						{
							EnterShort(tradeAmount);
							Print("Entered a short position with amount: " + tradeAmount);
						}
						else
						{
							Print("No action taken.");
						}
                    }
                    else
                    {
                        Print("POST request failed. Response status code: " + response.StatusCode);
						ExitPosition();
                    }
                }
                catch (Exception ex)
                {
                    Print("Exception occurred: " + ex.Message);
					ExitPosition();
                }
            }
        }
		
		private void ExitPosition()
		{
			ExitLong();
			ExitShort();
			Print("Exited all positions.");
		}
	}
}