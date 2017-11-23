using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot_Generator
{
    public class MandelbrotGPU
    {
        private static bool release = false;
        eLanguage gpu_language = eLanguage.OpenCL;
        eGPUType gpu_type = eGPUType.OpenCL;

        GPGPU _gpu;
        CudafyModule km;

        int res_x, res_y, split_x, start_index;
        uint[] dev_colors = null;
        uint[] dev_buffer, buffer;
        public MandelbrotGPU(int device_id, int res_x, int res_y, int split_x, uint[] buffer, int start_index)
        {
            CudafyTranslator.Language = gpu_language;

            if (!release)
            {
                km = CudafyTranslator.Cudafy(typeof(MandelbrotCode));
                km.Serialize("kernel.gpu");
            }
            else
            {
                km = CudafyModule.Deserialize("kernel.gpu");
            }

            _gpu = CudafyHost.GetDevice(gpu_type, device_id);
            var asd = _gpu.FreeMemory;
            _gpu.LoadModule(km);

            this.res_x = res_x;
            this.res_y = res_y;
            this.buffer = buffer;
            this.dev_buffer = _gpu.Allocate<uint>(res_x * res_y);
            _gpu.CopyToDevice<uint>(this.buffer, start_index, this.dev_buffer, 0, res_x * res_y);
            this.split_x = split_x;
            this.start_index = start_index;
        }

        public void SetNewColors(uint[] colors)
        {
            if (dev_colors != null)
                _gpu.Free(dev_colors);

            this.dev_colors = _gpu.CopyToDevice<uint>(colors);
        }

        public void UpdateArea(double x0, double x1, double y0, double y1, int max_iter)
        {
            for (int i = 0; i < split_x; i++)
            {
                _gpu.Launch(new dim3(res_y), new dim3(res_x / split_x)).calculate(x0, x1, y0, y1, dev_buffer, 
                    res_x, res_y, max_iter, dev_colors, res_x / split_x *  i);
            }  
        }
        public void Syncronize()
        {
            _gpu.CopyFromDevice<uint>(dev_buffer, 0, buffer, this.start_index, res_x * res_y);
            _gpu.Synchronize();
        }

        public void Close()
        {
            _gpu.FreeAll();
            _gpu.Dispose();
        }
    }
    class MandelbrotCode
    {
        [Cudafy]
        public static void calculate(GThread thread, double x0, double x1, double y0, double y1,
            uint[] buffer, int res_x, int res_y, int max_iter, uint[] colors, int split_x_offset)
        {
            double x2 = 0.0, y2 = 0.0, xi = 0.0, yi = 0.0;
            int it = 0;

            int y = thread.blockIdx.x;
            int x = thread.threadIdx.x + split_x_offset;

            double cx = x0 + x * (x1 - x0) / res_x;
            double cy = y0 + y * (y1 - y0)  / res_y;

            while(it < max_iter && x2 + y2 < 4.0)
            {
                yi = 2.0 * xi * yi + cy;
                xi = x2 - y2 + cx;

                x2 = xi * xi;
                y2 = yi * yi;

                it++;
            }

            buffer[y * res_x + x] = colors[it];
        }



        [Cudafy]
        public static int iterate(double cx, double cy, int max_iter)
        {
            double x2, y2;
            double xi = 0.0, yi = 0.0;
            int it;

            x2 = xi * xi;
            y2 = yi * yi;

            for (it = 0; it < max_iter && x2 + y2 < 4.0; it++)
            {
                yi = 2.0 * xi * yi + cy;
                xi = x2 - y2 + cx;

                x2 = xi * xi;
                y2 = yi * yi;
            }
            return it;
        }
    }
}
