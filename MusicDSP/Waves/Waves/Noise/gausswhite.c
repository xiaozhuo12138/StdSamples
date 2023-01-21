#define ranf() ((float) rand() / (float) RAND_MAX)

float ranfGauss (int m, float s)
{
   static int pass = 0;
   static float y2;
   float x1, x2, w, y1;

   if (pass)
   {
      y1 = y2;
   } else  {
      do {
         x1 = 2.0f * ranf () - 1.0f;
         x2 = 2.0f * ranf () - 1.0f;
         w = x1 * x1 + x2 * x2;
      } while (w >= 1.0f);

      w = (float)sqrt (-2.0 * log (w) / w);
      y1 = x1 * w;
      y2 = x2 * w;
   }
   pass = !pass;

   return ( (y1 * s + (float) m));
}
