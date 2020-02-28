using Cudafy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot_Generator
{
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

        
    }
}
