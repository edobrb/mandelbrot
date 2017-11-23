using Mandelbrot_Generator;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot_View
{
    class Settings
    {
        public int ResolutionX { get; set; }
        public int ResolutionY { get; set; }
        public int RenderResolutionX { get; set; }
        public int RenderResolutionY { get; set; }
        public bool FullScreen { get; set; }



        public List<int> Devices_OpenCL_ID { get; set; }
        public List<int> Devices_SplitX { get; set; }
        public List<int> Devices_PortionY { get; set; }

        public string MaxiterMode { get; set; }

        public List<MyColor> Colors { get; set; }
        public List<double> Weight { get; set; }
    }
}
