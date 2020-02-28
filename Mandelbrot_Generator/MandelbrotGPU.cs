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
            this.res_x = res_x;
            this.res_y = res_y;

            if (res_x * res_y != 0)
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
        }

        public void SetNewColors(uint[] colors)
        {
            if (res_x * res_y != 0)
            {
                if (dev_colors != null)
                    _gpu.Free(dev_colors);

                this.dev_colors = _gpu.CopyToDevice<uint>(colors);
            }
        }

        public void UpdateArea(double x0, double x1, double y0, double y1, int max_iter)
        {
            if (res_x * res_y != 0)
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
        }
        public void Syncronize()
        {
            if (res_x * res_y != 0)
            {
                _gpu.CopyFromDevice<uint>(dev_buffer, 0, buffer, this.start_index, res_x * res_y);
                _gpu.Synchronize();
            }
        }

        public void Close()
        {
            if (res_x * res_y != 0)
            {
                _gpu.Free(dev_buffer);
                _gpu.Free(dev_xi_a);
                _gpu.Free(dev_yi_a);
                _gpu.Free(dev_cx_a);
                _gpu.Free(dev_cy_a);
                _gpu.Free(dev_it_a);
                _gpu.Dispose();
            }
        }
    }
    
}
