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

        int res_x, res_y, split_y;
        uint[] dev_colors = null;
        uint[] dev_buffer, buffer;
        public MandelbrotGPU(int device_id, int res_x, int res_y, int split_y)
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
            this.buffer = new uint[res_x * res_y];
            this.dev_buffer = _gpu.CopyToDevice<uint>(this.buffer);
            this.split_y = split_y;
        }

        public void SetNewColors(uint[] colors)
        {
            if (dev_colors != null)
                _gpu.Free(dev_colors);

            this.dev_colors = _gpu.CopyToDevice<uint>(colors);
        }

        public uint[] GetArea(double x0, double x1, double y0, double y1, int max_iter)
        {
            for (int i = 0; i < split_y; i++)
            {
                _gpu.Launch(new dim3(res_y), new dim3(res_x / split_y)).calculate(x0, x1, y0, y1, dev_buffer, res_x, res_y, max_iter, dev_colors, split_y, i);
            }
            _gpu.CopyFromDevice<uint>(dev_buffer, buffer);
            _gpu.Synchronize();
            return buffer;
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
            uint[] buffer, int res_x, int res_y, int max_iter, uint[] colors, int split_y_count, int split_y_region)
        {
            double x2 = 0.0, y2 = 0.0;
            double xi = 0.0, yi = 0.0;
            int it;

            int y = thread.blockIdx.x;
            int x = thread.threadIdx.x + res_x * split_y_region / split_y_count;

            double cx = x0 + x * (x1 - x0) / res_x;
            double cy = y0 + y * (y1 - y0)  / res_y;



            it = 0;
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
