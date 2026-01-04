package gss;

public class Mode
{
	private static boolean strictBroadcast;

	public static void setStrictBroadcast(boolean enable)
	{
		strictBroadcast = enable;
	}
	public static boolean isStrictBroadcastEnabled()
	{
		return strictBroadcast;
	}
}
