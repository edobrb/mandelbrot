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

        int res_x, res_y, split_x, start_index, maxiter_per_step;
        uint[] dev_colors = null;
        uint[] dev_buffer, buffer;

        double[] dev_xi_a;
        double[] dev_yi_a;
        double[] dev_cx_a;
        double[] dev_cy_a;
        int[] dev_it_a;

        public MandelbrotGPU(int device_id, int res_x, int res_y, int split_x, uint[] buffer, int start_index, int maxiter_per_step)
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
            this.maxiter_per_step = maxiter_per_step;

            this.dev_xi_a = _gpu.Allocate<double>(res_x * res_y / split_x);
            this.dev_yi_a = _gpu.Allocate<double>(res_x * res_y / split_x);
            this.dev_cx_a = _gpu.Allocate<double>(res_x * res_y / split_x);
            this.dev_cy_a = _gpu.Allocate<double>(res_x * res_y / split_x);
            this.dev_it_a = _gpu.Allocate<int>(res_x * res_y / split_x);
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
                if (maxiter_per_step >= max_iter || maxiter_per_step <= 0)
                {
                    _gpu.Launch(new dim3(res_y), new dim3(res_x / split_x)).calculate(x0, x1, y0, y1, dev_buffer,
                        res_x, res_y, max_iter, dev_colors, res_x / split_x * i);
                }
                else
                {
                    _gpu.Launch(new dim3(res_y), new dim3(res_x / split_x)).initialize(x0, x1, y0, y1, dev_buffer,
                        res_x, res_y, max_iter, dev_colors, res_x / split_x * i,
                        dev_xi_a, dev_yi_a, dev_it_a, dev_cx_a, dev_cy_a);
                    _gpu.Synchronize();
                    int mi = max_iter;
                    while (mi > 0)
                    {
                        _gpu.Launch(new dim3(res_y), new dim3(res_x / split_x)).iterate(dev_xi_a, dev_yi_a, dev_it_a, dev_cx_a, dev_cy_a,
                            maxiter_per_step <= 0 ? max_iter : (mi > maxiter_per_step ? maxiter_per_step : mi), res_y);

                        if (maxiter_per_step <= 0) break;
                        mi -= maxiter_per_step;
                    }
                    _gpu.Synchronize();
                    _gpu.Launch(new dim3(res_y), new dim3(res_x / split_x)).finalize(res_x, res_y, res_x / split_x * i, dev_buffer, dev_colors, dev_it_a);
                }
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
        public static void initialize(GThread thread, double x0, double x1, double y0, double y1,
            uint[] buffer, int res_x, int res_y, int max_iter, uint[] colors, int split_x_offset,
            double[] xi_a, double[] yi_a, int[] it_a, double[] cx_a, double[] cy_a)
        {
            int y = thread.blockIdx.x;
            int x = thread.threadIdx.x + split_x_offset;
            int tid = thread.blockIdx.x + res_y * thread.threadIdx.x;

            cx_a[tid] = x0 + x * (x1 - x0) / res_x;
            cy_a[tid] = y0 + y * (y1 - y0) / res_y;

            xi_a[tid] = 0;
            yi_a[tid] = 0;
            it_a[tid] = 0;
        }

        [Cudafy]
        public static void iterate(GThread thread, double[] xi_a, double[] yi_a, int[] it_a, double[] cx_a, double[] cy_a,
            int max_iter, int res_y)
        {
            int tid = thread.blockIdx.x + res_y * thread.threadIdx.x;

            int k = 0;
            int it = it_a[tid];
            double xi = xi_a[tid];
            double yi = yi_a[tid];
            double cx = cx_a[tid];
            double cy = cy_a[tid];

            double x2 = xi * xi, y2 = yi * yi;
            while (k < max_iter && x2 + y2 < 4.0)
            {
                yi = 2.0 * xi * yi + cy;
                xi = x2 - y2 + cx;

                x2 = xi * xi;
                y2 = yi * yi;

                it++;
                k++;
            }

            xi_a[tid] = xi;
            yi_a[tid] = yi;
            it_a[tid] = it;
        }

        [Cudafy]
        public static void finalize(GThread thread, int res_x, int res_y, int split_x_offset, uint[] buffer, uint[] colors, int[] it_a)
        {
            int y = thread.blockIdx.x;
            int x = thread.threadIdx.x + split_x_offset;
            int tid = thread.blockIdx.x + res_y * thread.threadIdx.x;
            buffer[y * res_x + x] = colors[it_a[tid]];
        }

        [Cudafy]
        public static void calculate(GThread thread, double x0, double x1, double y0, double y1,
            uint[] buffer, int res_x, int res_y, int max_iter, uint[] colors, int split_x_offset)
        {
            double x2 = 0.0, y2 = 0.0, xi = 0.0, yi = 0.0;
            int it = 0;

            int y = thread.blockIdx.x;
            int x = thread.threadIdx.x + split_x_offset;

            double cx = x0 + x * (x1 - x0) / res_x;
            double cy = y0 + y * (y1 - y0) / res_y;

            while (it < max_iter && x2 + y2 < 4.0)
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
        public static void calculate1(GThread thread, double x0, double x1, double y0, double y1,
           uint[] buffer, int res_x, int res_y, int max_iter, uint[] colors, int split_x_offset)
        {
            double xtmp = 0.0, ytmp = 0.0, xi = 0.0, yi = 0.0;
            int it = 0;

            int y = thread.blockIdx.x;
            int x = thread.threadIdx.x + split_x_offset;

            double cx = x0 + x * (x1 - x0) / res_x;
            double cy = y0 + y * (y1 - y0) / res_y;

            while (it < max_iter && (xtmp * xtmp) + (ytmp * ytmp) < 4.0)
            {

                xi = it2_x(xtmp, ytmp) + cx;
                yi = it2_y(xtmp, ytmp) + cy;

                xtmp = xi;
                ytmp = yi;

                it++;
            }

            buffer[y * res_x + x] = colors[it];
        }
        [Cudafy]
        public static double it2_x(double x, double y)
        {
            return x * x - y * y;
        }
        [Cudafy]
        public static double it2_y(double x, double y)
        {
            return 2 * x * y;
        }
        [Cudafy]
        public static double it3_x(double x, double y)
        {
            return x * x * x -  3 * x * y * y;
        }
        [Cudafy]
        public static double it3_y(double x, double y)
        {
            return 3 * x * x * y - y * y * y;
        }
        [Cudafy]
        public static double it4_x(double x, double y)
        {
            return x * x * x * x - 6 * x * x * y * y + y * y * y * y;
        }
        [Cudafy]
        public static double it4_y(double x, double y)
        {
            return 4 * x * x * x * y - 4 * x * y * y * y;
        }
    }
}
