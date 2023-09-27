
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
		private int numBar;
		private int sequenceLength = 20;
		

		public class RequestData
        {
            public List<double> open = new List<double>();
            public List<double> high = new List<double>();
            public List<double> low = new List<double>();
            public List<double> close = new List<double>();
            public List<double> volume = new List<double>();
			public int position;
            public double balance;
            public double entryPrice;
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
			}
		}
		
		
		protected override async void OnBarUpdate()
		{
		    if (State == State.Historical)
                return;	

			if(CurrentBar > sequenceLength)
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
				data.open.Add(Open[i]);
				data.high.Add(High[i]);
				data.low.Add(Low[i]);
				data.close.Add(Close[i]);
				data.volume.Add(Volume[i]);
			}
			
			data.position = GetPosition();
			data.entryPrice = Position.AveragePrice;
			data.balance = GetAccountBalance();
		}
		
		private void ClearData()
		{
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
		
		private int GetPosition()
		{
			switch(Position.MarketPosition) 
			{
			  case MarketPosition.Flat:
			    return 0;
			  case MarketPosition.Long:
			    return 1;
			  case MarketPosition.Short:
			    return 2;
			  default:
			    return 0;
			}
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
		                int prediction = parsedResponse.prediction;

						PerformAction(prediction);
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
		
		private void PerformAction(int prediction)
		{
			int position = GetPosition();
			
			if (prediction != position)
			{
				ExitPosition();
				if (prediction == 1)
				{	
				  EnterLong();
				  Print("Entered long with amount: 1");
				}
				else if (prediction == 2)
				{	
				  EnterShort();
				  Print("Entered short with amount: 1");
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