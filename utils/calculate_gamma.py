import click

@click.command()

@click.option('--img_width', help='Train images width', required=True, type=click.INT)
@click.option('--img_height', help='Train images height', required=True, type=click.INT)
@click.option('--mini_batch_size', help='Train minibatch', required=True, type=click.INT)

def calculate_gamma (img_width:int, img_height:int, mini_batch_size:int):
  '''
  Calculates gamma search range for given img dimensions and minibatch size
  '''
  
  gamma_0 = 0.0002 * img_width * img_height / mini_batch_size
  print(f"gamma_0 is {gamma_0}. Do a grid search in the range: [ {gamma_0/5} | {gamma_0*5} ]")



#----------------------------------------------------------------------------

if __name__ == "__main__":
  calculate_gamma()

#----------------------------------------------------------------------------

