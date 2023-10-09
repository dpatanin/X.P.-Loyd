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
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using SharpDX;
#endregion

//This namespace holds Indicators in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Indicators
{
	public class SigmoidGate : Indicator
	{
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description							= @"Receives an array of input series as outputs from sigmoid indicators.";
				Name								= "SigmoidGate";
				Calculate							= Calculate.OnEachTick;
				DrawOnPricePanel					= false;
				IsSuspendedWhileInactive			= false;
				BarsRequiredToPlot					= 1;
				
				Threshold = 0.9;
				AddPlot(Brushes.Turquoise, "Signal");
			}
			if (State == State.DataLoaded)
			{
				Draw.HorizontalLine(this, "Zero", 0, Brushes.WhiteSmoke);
			}
		}

		protected override void OnBarUpdate()
		{
			int lowerCount = 0;
			int upperCount = 0;
			
			foreach (Series<double> signal in Signals)
			{
				if (signal[0] < -Threshold)
					lowerCount++;
				else if (signal[0] > Threshold)
					upperCount++;
			}
			
			int output = 0;
			if (upperCount > lowerCount)
				output = 1;
			else if (lowerCount > upperCount)
				output = -1;
			
			Default[0] = output;
		}

		#region Properties
		[NinjaScriptProperty]
		[Display(Name = "Signals", GroupName = "Parameters", Order = 0)]
		public List<ISeries<double>> Signals
		{ get; set; }
		
		[Range(0, 1), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 1)]
		public double Threshold
		{ get; set; }
		
		[Browsable(false)]
		[XmlIgnore()]
		public Series<double> Default
		{
			get { return Values[0]; }
		}
		#endregion
	
		}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private SigmoidGate[] cacheSigmoidGate;
		public SigmoidGate SigmoidGate(List<ISeries<double>> signals, double threshold)
		{
			return SigmoidGate(Input, signals, threshold);
		}

		public SigmoidGate SigmoidGate(ISeries<double> input, List<ISeries<double>> signals, double threshold)
		{
			if (cacheSigmoidGate != null)
				for (int idx = 0; idx < cacheSigmoidGate.Length; idx++)
					if (cacheSigmoidGate[idx] != null && cacheSigmoidGate[idx].Signals == signals && cacheSigmoidGate[idx].Threshold == threshold && cacheSigmoidGate[idx].EqualsInput(input))
						return cacheSigmoidGate[idx];
			return CacheIndicator<SigmoidGate>(new SigmoidGate(){ Signals = signals, Threshold = threshold }, input, ref cacheSigmoidGate);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.SigmoidGate SigmoidGate(List<ISeries<double>> signals, double threshold)
		{
			return indicator.SigmoidGate(Input, signals, threshold);
		}

		public Indicators.SigmoidGate SigmoidGate(ISeries<double> input , List<ISeries<double>> signals, double threshold)
		{
			return indicator.SigmoidGate(input, signals, threshold);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.SigmoidGate SigmoidGate(List<ISeries<double>> signals, double threshold)
		{
			return indicator.SigmoidGate(Input, signals, threshold);
		}

		public Indicators.SigmoidGate SigmoidGate(ISeries<double> input , List<ISeries<double>> signals, double threshold)
		{
			return indicator.SigmoidGate(input, signals, threshold);
		}
	}
}

#endregion
