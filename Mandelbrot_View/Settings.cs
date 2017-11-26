using Mandelbrot_Generator;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot_View
{
    enum RenderMode
    {
        Manual,//when user ask for render
        Forced,//every new frame is rendered
        Fluid,//whenever is possible
    }
    enum MaxiterMode
    {
       Static,
       Dynamic,
    }
    class Settings
    {
        public double InitialCenterX { get; set; }
        public double InitialCenterY { get; set; }
        public double InitialViewportSizeY { get; set; }
        public double InitialMaxIter { get; set; }

        public int ResolutionX { get; set; }
        public int ResolutionY { get; set; }
        public int RenderResolutionX { get; set; }
        public int RenderResolutionY { get; set; }
        public bool FullScreen { get; set; }


        public List<int> Devices_OpenCL_ID { get; set; }
        public List<int> Devices_SplitX { get; set; }
        public List<int> Devices_PortionY { get; set; }
        public List<int> Maxiter_Per_Step { get; set; }

        [JsonConverter(typeof(StringEnumConverter))]
        public MaxiterMode MaxiterMode { get; set; }
        public string MaxIterDynamicFunction { get; set; }
        [JsonConverter(typeof(StringEnumConverter))]
        public RenderMode RenderMode { get; set; }


        public List<MyColor> Colors { get; set; }
        public List<double> Weight { get; set; }
        public string GradientFunction { get; set; }
    }
}
