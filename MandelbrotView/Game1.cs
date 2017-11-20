using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System;

namespace MandelbrotView
{
    /// <summary>
    /// This is the main type for your game.
    /// </summary>
    public class Game1 : Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;
        int res_x = 1920;
        int res_y = 1080;
        int maxiter = 100;
        byte[] buffer;
        byte[] colors;
        bool toUpdate;
        double x0, x1, y0, y1;

        Texture2D screen;

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            this.graphics.PreferredBackBufferWidth = res_x;
            this.graphics.PreferredBackBufferHeight = res_y;
            this.TargetElapsedTime = TimeSpan.FromSeconds(1.0 / 60.0);
            this.graphics.IsFullScreen = true;
        }

        private Color GetLinearGradient(double v, double minV, double maxV, Color[] colors, double[] range)
        {
            byte[] cols = new byte[colors.Length * 3];
            for (int i = 0; i < colors.Length; i++)
            {
                cols[i * 3 + 0] = colors[i].R;
                cols[i * 3 + 1] = colors[i].G;
                cols[i * 3 + 2] = colors[i].B;
            }
            byte[] c = Mandelbrot.Mandelbrot.GetLinearGradient(v, minV, maxV, cols, range);
            return new Color(c[0], c[1], c[2]);
        }
        protected override void Initialize()
        {
            buffer = new byte[res_x * res_y * 4];
            RecalcColor();
            toUpdate = true;
            x0 = -res_x / 100;
            x1 = res_x / 100;
            y0 = -res_y / 100;
            y1 = res_y / 100;

           
            oldkey = Keyboard.GetState();
            base.Initialize();
        }
        private void RecalcColor()
        {
            colors = new byte[(maxiter + 1) * 3];
            for (int i = 0; i < maxiter + 1; i++)
            {
                
                Color c = GetLinearGradient(i, 0, maxiter,
                    new Color[] { Color.DarkGray, Color.DarkGray, Color.Black, Color.DarkRed, Color.Red, Color.DarkRed, Color.Black },
                    new double[] { 0.2, 1, 3, 3, 3, 0.5 });

                /*Color c = GetLinearGradient(i, 0, maxiter,
                    new Color[] { Color.DarkGray, Color.Red, Color.Black },
                    new double[] { 1, 1 });*/
                colors[i * 3 + 0] = c.R;
                colors[i * 3 + 1] = c.G;
                colors[i * 3 + 2] = c.B;
            }
        }

        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);
            screen = new Texture2D(this.GraphicsDevice, res_x, res_y);
        }

        KeyboardState oldkey;
        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();

           

           

            base.Update(gameTime);
        }

        private void Input()
        {
            KeyboardState key = Keyboard.GetState();
            if (key.IsKeyDown(Keys.W) && oldkey.IsKeyUp(Keys.W))
            {
                x0 += (x1 - x0) / 10;
                x1 -= (x1 - x0) / 10;

                y0 += (y1 - y0) / 10;
                y1 -= (y1 - y0) / 10;

                double scale = res_x / (x1 - x0);
                maxiter = (int)(Math.Sqrt(2 * Math.Sqrt(Math.Abs(1 - Math.Sqrt(5 * scale)))) * 66.5);
                //maxiter = (int)(50 * Math.Pow(Math.Log10(res_x / (x1 - x0)), 1.25));
                RecalcColor();
                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.S) && oldkey.IsKeyUp(Keys.S))
            {
                x0 -= (x1 - x0) / 10;
                x1 += (x1 - x0) / 10;

                y0 -= (y1 - y0) / 10;
                y1 += (y1 - y0) / 10;

                double scale = res_x / (x1 - x0);
                maxiter = (int)(Math.Sqrt(2 * Math.Sqrt(Math.Abs(1 - Math.Sqrt(5 * scale)))) * 66.5);
                RecalcColor();
                toUpdate = true;
            }

            if (key.IsKeyDown(Keys.Left) && oldkey.IsKeyUp(Keys.Left))
            {
                x0 -= (x1 - x0) / 30;
                x1 -= (x1 - x0) / 30;

                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.Right) && oldkey.IsKeyUp(Keys.Right))
            {
                x0 += (x1 - x0) / 30;
                x1 += (x1 - x0) / 30;

                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.Up) && oldkey.IsKeyUp(Keys.Up))
            {
                y0 -= (y1 - y0) / 30;
                y1 -= (y1 - y0) / 30;

                toUpdate = true;
            }
            if (key.IsKeyDown(Keys.Down) && oldkey.IsKeyUp(Keys.Down))
            {
                y0 += (y1 - y0) / 30;
                y1 += (y1 - y0) / 30;

                toUpdate = true;
            }

            //oldkey = key;
        }
        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            Input();
            if (toUpdate)
            {
                Mandelbrot.Mandelbrot.GetMandelbrotRegion(x0, x1, y0, y1, buffer, res_x, res_y, maxiter, colors);
                screen.SetData<byte>(buffer);
                toUpdate = false;
            }

            GraphicsDevice.Clear(Color.CornflowerBlue);

            spriteBatch.Begin();
            spriteBatch.Draw(screen, new Rectangle(0, 0, res_x, res_y), Color.White);
            spriteBatch.End();

            base.Draw(gameTime);
        }
    }
}
