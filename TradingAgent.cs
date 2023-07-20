#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Strategies
{
	using System;
	using System.Net.Http;
	using System.Threading.Tasks;
	
	public class TradingAgent : Strategy
	{	
		private DateTime firstDateTime;
		private DateTime lastDateTime;
		private bool isExecuted;
		private bool isLastSequence;
		private int numBar;
		private int sequenceLength = 10;
		
		private List<double> progressList = new List<double>();
		private List<double> openList = new List<double>();
		private List<double> highList = new List<double>();
		private List<double> lowList = new List<double>();
		private List<double> closeList = new List<double>();
		private List<double> volumeList = new List<double>();
		private int contracts;
        private double entryPrice;
        private double balance;
		
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
				StartBehavior								= StartBehavior.WaitUntilFlat;
				TimeInForce									= TimeInForce.Gtc;
				TraceOrders									= false;
				RealtimeErrorHandling						= RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling							= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade							= 0;
				// Disable this property for performance gains in Strategy Analyzer optimizations
				// See the Help Guide for additional information
				IsInstantiatedOnEachOptimizationIteration	= true;
			}
		}
		protected override async void OnBarUpdate()
		{	
			numBar = CurrentBar;
			if(!isExecuted)
			{
				firstDateTime = Time[0];
				lastDateTime = Time[0].Date.AddHours(18);
				isExecuted = true;
			}
			
			CalcProgress();
			
			if((numBar % sequenceLength == 0 && numBar != 0) || isLastSequence)
			{
				CollectData();
				//SendHttpRequest();
				ClearData();
			}
		}
		
		private void CollectData()
		{			
			for (int i = 0; i < sequenceLength; i++)
			{
				openList.Add(Open[i]);
				highList.Add(High[i]);
				lowList.Add(Low[i]);
				closeList.Add(Close[i]);
				volumeList.Add(Volume[i]);
			}
			contracts = Position.Quantity;
			entryPrice = Position.AveragePrice;
			balance = GetAccountBalance();
		}
		
		private void ClearData()
		{
			progressList.Clear();
			openList.Clear();
			highList.Clear();
			lowList.Clear();
			closeList.Clear();
			volumeList.Clear();
		}
		
		private double GetAccountBalance()
		{
			Account a = Account.All.First(t => t.Name == "Sim101");
			double value = a.Get(AccountItem.CashValue, Currency.UsDollar);
			return value;
		}
		
		//Calculate the current progress
		private void CalcProgress()
		{
			double firstHours = (firstDateTime - firstDateTime.Date).TotalHours;
			double lastHours = (lastDateTime - lastDateTime.Date).TotalHours;
			double hourDiff = lastHours - firstHours;
			progressList.Add(numBar / (hourDiff * 60));
			
			if(progressList.Last() == 1)
			{
				ResetParameters();
			}
		}
		
		//Resets Parameters because new Trading Day begins
		private void ResetParameters()
		{
			numBar = 0;
			isExecuted = false;
			isLastSequence = true;
		}
		
		private async void SendHttpRequest()
		{			
			 // Create an HttpClient instance
	        using (HttpClient client = new HttpClient())
	        {
	            try
	            {
					// Create a request body as string
	                string requestBody ="";
	                
					// Create an HttpContent instance from the request body
	                HttpContent content = new StringContent(requestBody);
					
	                // Send GET request to a URL
	                string url = "https://google.de/";
	                HttpResponseMessage response = await client.PostAsync(url, content);

					/*
	                // Check if the request was successful
	                if (response.IsSuccessStatusCode)
	                {
						
	                    // Read the response content as string
	                    string responseBody = await response.Content.ReadAsStringAsync();

	                    // Do something with the response data
	                    Print("Response received:");
	                    Print(responseBody);
	                }
	                else
	                {
	                    // Request was not successful, handle the error
	                    Print("Request failed with status code: " + response.StatusCode);
	                }
					*/
	            }
	            catch (Exception ex)
	            {
	                // An exception occurred, handle the error
	                Print("An error occurred: " + ex.Message);
	            }
	        }
		}
		
		#region Properties
		
		#endregion
	}
}