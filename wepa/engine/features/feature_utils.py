import pandas as pd
import numpy


## Other impliementations below that use sigmoids and distributions.
## just go with a simple linear weighting

def tail_scale(df, prob_col, weight_array):
    '''
    Weights tails of a distribution for a probability column
    '''
    ##  normalize the direction of the tail ##
    tail_normalized = numpy.where(
        df[prob_col].fillna(.5) < .5,
        df[prob_col].fillna(.5),
        1-df[prob_col].fillna(.5)
    )
    ## avoid divide by sero instances
    avoid_zero = numpy.where(
        weight_array==0,
        99, ## if no weight, this will be ignored. Just dont trigger div 0 error
        weight_array / 2 ## since weight array is 0 to 1 but tail is 0 to 0.5
    )
    ## calc ##
    ## This sets the EPA wight to 0 when tail normalized is 0
    ## and increases it linearly to 1 at prob_col=weight
    vals = numpy.minimum(1, tail_normalized * (1 / avoid_zero))
    ## return ##
    ## if the discount is 0, we return 0, else the scaled values
    return numpy.where(
        weight_array == 0,
        0,
        vals
    )

# def tail_scale(df, prob_col, weight_array):
#     '''
#     A dynamic super gaussian that always produces a distribtion where:
#          > f(0.0) = 0
#          > f(0.5) = 1
#          > f(1.0) = 0
#     Distribution width, definid by standard deviation (from (0,0.5)), controls how fat
#     the tails are. A high standard deviation creates fatter tails
#     '''
#     def super_gaussian(probs, sdev):
#         '''
#         calculates the super gaussian value. This formula uses sdev to
#         derive a P that ensures the 0 to 1 domain is maintained. As sdev
#         approaches the mean (0.5), P approaches infinity
#         '''
#         ## handle invalid ranges for sdev ##
#         sdev = numpy.minimum(0.49, numpy.maximum(0.01, sdev))
#         return numpy.exp(
#             - (
#                 (
#                     numpy.absolute(
#                         probs - 0.5
#                     ) ** (
#                         1 / (0.5 - sdev)
#                     )
#                 ) / (
#                     2 * (sdev ** (
#                         1 / (0.5 - sdev)
#                     ))
#                 )
#             )
#         )
#     ## since wepa weights can range from -1 to 2, they must be normalized to 
#     ## the 0,.5 scale. Note, if a positive number is passed it means the tails
#     ## are being more highly weighted. This sign change happens post gaussian calc
#     normalized_deviation = numpy.where(
#         weight_array <= 0,
#         0.5 - (1 + weight_array) / 2,
#         0.5 - (2 - weight_array) / 4
#     )
#     ## calculate, and exclude values that will cause divide by 0 or overflows ##
#     return numpy.where(
#         weight_array == 0,
#         0,
#         numpy.where(
#             weight_array > 0,
#             ## sign flip ##
#             1 - super_gaussian(df[prob_col], normalized_deviation),
#             super_gaussian(df[prob_col], normalized_deviation) - 1
#         )
#     )


# def tail_scale(df, prob_col, weight_array):
#     '''
#     Takes a dataframe, a column of probabilities, and a weight
#     to return a discounting or boosting along an s curve at the tails
#     where abs(weight) percent of the curve experiences an adjustment
#     For instance, for a weight = -0.8, 80% of the distribution (40% for each tail)
#     would  experience a negative adjustment:
#         > prob = 0%, adjustment = -100%
#         > prob = 20%, adjustment = -50%
#         > prob = 40%, adjustment = 0%
#         > prob = 60%, adjustment = 0%
#         > prob = 80%, adjustment = -50%
#         > prob = 100%, adjustment = -100% 
#     '''
#     ## scaling factor ##
#     factor = 2
#     ## formulate the basic shape ##
#     ## make the tails on the same side, ie .05 == .95
#     tail_normalized = numpy.where(
#         df[prob_col].fillna(.5) < .5,
#         df[prob_col].fillna(.5),
#         1-df[prob_col].fillna(.5)
#     ) + 0.0001
#     ## get the start point of the drop ##
#     start_point = numpy.absolute(weight_array) / 2
#     ## form the shape ##
#     shape = numpy.where(
#         (tail_normalized < start_point) & (start_point>0.001) ,
#         1 / (1 + (((tail_normalized / start_point) / (1 - (tail_normalized / start_point))) ** (-1 * factor))),
#         1
#     )
#     ## set the direction ##
#     directed_shape = numpy.where(
#         weight_array < 0,
#         shape - 1,
#         1 - shape
#     )
#     ## return ##
#     return directed_shape
