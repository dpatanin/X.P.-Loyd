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
	public class RSIwithStopAndTarget : Strategy
	{
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"RSI with Stop Loss and Profit Target";
				Name										= "RSIwithStopAndTarget";
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
				BarsRequiredToTrade							= 20;
				// Disable this property for performance gains in Strategy Analyzer optimizations
				// See the Help Guide for additional information
				IsInstantiatedOnEachOptimizationIteration	= true;
				RSIPeriod					= 14;
				RSISmooth					= 3;
				ProfitTarget					= 12;
				StopLoss					= 6;
			}
			else if (State == State.Configure)
			{
				AddChartIndicator(RSI(RSIPeriod, RSISmooth));
     
    			SetStopLoss(CalculationMode.Ticks, StopLoss);
    			SetProfitTarget(CalculationMode.Ticks, ProfitTarget);
			}
		}

		protected override void OnBarUpdate()
		{
			if (CurrentBar < RSIPeriod)
    		return;
 
			if(CrossAbove(RSI(RSIPeriod, RSISmooth), 20, 1))
			EnterLong();
		}

		#region Properties
		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="RSIPeriod", Description="RSI Period", Order=1, GroupName="Parameters")]
		public int RSIPeriod
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="RSISmooth", Description="RSI Smooth", Order=2, GroupName="Parameters")]
		public int RSISmooth
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="ProfitTarget", Description="ProfitTarget", Order=3, GroupName="Parameters")]
		public int ProfitTarget
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="StopLoss", Order=4, GroupName="Parameters")]
		public int StopLoss
		{ get; set; }
		#endregion

	}
}
